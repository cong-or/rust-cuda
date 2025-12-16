use cust::prelude::*;
use std::error::Error;
use std::time::Instant;

pub struct SaxpyBenchmark {
    n: usize,
    a: f32,
    x: Vec<f32>,
    y: Vec<f32>,
}

impl SaxpyBenchmark {
    pub fn new(n: usize) -> Self {
        // Initialize with random-ish data
        let x: Vec<f32> = (0..n).map(|i| (i % 100) as f32 / 100.0).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 50) as f32 / 50.0).collect();

        Self {
            n,
            a: 2.5,
            x,
            y,
        }
    }

    pub fn run_rust_cuda(
        &self,
        module: &Module,
        stream: &Stream,
        iterations: usize,
    ) -> Result<f64, Box<dyn Error>> {
        // Allocate GPU memory
        let x_gpu = self.x.as_slice().as_dbuf()?;
        let mut y_gpu = self.y.as_slice().as_dbuf()?;

        // Get kernel
        let saxpy = module.get_function("saxpy")?;

        // Calculate launch configuration
        let block_size = 256;
        let grid_size = (self.n as u32 + block_size - 1) / block_size;

        // Warmup
        for _ in 0..10 {
            unsafe {
                launch!(
                    saxpy<<<grid_size, block_size, 0, stream>>>(
                        self.n,
                        self.a,
                        x_gpu.as_device_ptr(),
                        x_gpu.len(),
                        y_gpu.as_device_ptr(),
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
                    saxpy<<<grid_size, block_size, 0, stream>>>(
                        self.n,
                        self.a,
                        x_gpu.as_device_ptr(),
                        x_gpu.len(),
                        y_gpu.as_device_ptr(),
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
        // SAXPY does: read X, read Y, write Y = 3 * n * sizeof(f32) bytes
        let bytes = 3.0 * self.n as f64 * 4.0;
        let time_s = time_ms / 1000.0;
        bytes / time_s / 1e9
    }

    pub fn calculate_gflops(&self, time_ms: f64) -> f64 {
        // SAXPY: 2 operations per element (multiply + add)
        let flops = 2.0 * self.n as f64;
        let time_s = time_ms / 1000.0;
        flops / time_s / 1e9
    }
}
