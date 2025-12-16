use cust::prelude::*;
use std::error::Error;
use std::time::Instant;

pub struct GemmBenchmark {
    m: usize,
    n: usize,
    k: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
}

impl GemmBenchmark {
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32 / 100.0).collect();
        let c: Vec<f32> = vec![0.0; m * n];

        Self { m, n, k, a, b, c }
    }

    pub fn run_rust_cuda_naive(
        &self,
        module: &Module,
        stream: &Stream,
        iterations: usize,
    ) -> Result<f64, Box<dyn Error>> {
        self.run_kernel(module, stream, "gemm_naive", (16, 16), iterations)
    }

    pub fn run_rust_cuda_tiled(
        &self,
        module: &Module,
        stream: &Stream,
        iterations: usize,
    ) -> Result<f64, Box<dyn Error>> {
        self.run_kernel(module, stream, "gemm_tiled", (16, 16), iterations)
    }

    fn run_kernel(
        &self,
        module: &Module,
        stream: &Stream,
        kernel_name: &str,
        block_dim: (u32, u32),
        iterations: usize,
    ) -> Result<f64, Box<dyn Error>> {
        let a_gpu = self.a.as_slice().as_dbuf()?;
        let b_gpu = self.b.as_slice().as_dbuf()?;
        let c_gpu = self.c.as_slice().as_dbuf()?;

        let kernel = module.get_function(kernel_name)?;

        let (block_x, block_y) = block_dim;
        let grid_x = (self.m as u32 + block_x - 1) / block_x;
        let grid_y = (self.n as u32 + block_y - 1) / block_y;

        // Warmup
        for _ in 0..5 {
            unsafe {
                launch!(
                    kernel<<<(grid_x, grid_y), (block_x, block_y), 0, stream>>>(
                        self.m,
                        self.n,
                        self.k,
                        a_gpu.as_device_ptr(),
                        a_gpu.len(),
                        b_gpu.as_device_ptr(),
                        b_gpu.len(),
                        c_gpu.as_device_ptr(),
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
                    kernel<<<(grid_x, grid_y), (block_x, block_y), 0, stream>>>(
                        self.m,
                        self.n,
                        self.k,
                        a_gpu.as_device_ptr(),
                        a_gpu.len(),
                        b_gpu.as_device_ptr(),
                        b_gpu.len(),
                        c_gpu.as_device_ptr(),
                    )
                )?;
            }
        }
        stream.synchronize()?;
        let elapsed = start.elapsed();

        let avg_time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        Ok(avg_time_ms)
    }

    pub fn calculate_gflops(&self, time_ms: f64) -> f64 {
        // GEMM: 2*M*N*K operations (multiply-add for each element)
        let flops = 2.0 * self.m as f64 * self.n as f64 * self.k as f64;
        let time_s = time_ms / 1000.0;
        flops / time_s / 1e9
    }
}
