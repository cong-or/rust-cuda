use cust::prelude::*;
use std::error::Error;
use std::time::Instant;

pub struct ReductionBenchmark {
    n: usize,
    data: Vec<f32>,
}

impl ReductionBenchmark {
    pub fn new(n: usize) -> Self {
        let data: Vec<f32> = (0..n).map(|i| (i % 100) as f32 / 100.0).collect();
        Self { n, data }
    }

    pub fn run_rust_cuda(
        &self,
        module: &Module,
        stream: &Stream,
        iterations: usize,
    ) -> Result<f64, Box<dyn Error>> {
        let input_gpu = self.data.as_slice().as_dbuf()?;

        let block_size = 256;
        let grid_size = (self.n as u32 + block_size - 1) / block_size;

        // Output buffer for partial sums
        let output_vec = vec![0.0f32; grid_size as usize];
        let output_gpu = output_vec.as_slice().as_dbuf()?;

        let kernel = module.get_function("reduction_sum")?;

        // Warmup
        for _ in 0..10 {
            unsafe {
                launch!(
                    kernel<<<grid_size, block_size, 0, stream>>>(
                        input_gpu.as_device_ptr(),
                        input_gpu.len(),
                        output_gpu.as_device_ptr(),
                        self.n,
                    )
                )?;
            }
        }
        stream.synchronize()?;

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                launch!(
                    kernel<<<grid_size, block_size, 0, stream>>>(
                        input_gpu.as_device_ptr(),
                        input_gpu.len(),
                        output_gpu.as_device_ptr(),
                        self.n,
                    )
                )?;
            }
        }
        stream.synchronize()?;
        let elapsed = start.elapsed();

        let avg_time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        Ok(avg_time_ms)
    }

    pub fn calculate_bandwidth_gbps(&self, time_ms: f64) -> f64 {
        // Reduction reads all elements once
        let bytes = self.n as f64 * 4.0;
        let time_s = time_ms / 1000.0;
        bytes / time_s / 1e9
    }
}
