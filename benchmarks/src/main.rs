use cust::prelude::*;
use std::error::Error;

mod saxpy_bench;
mod gemm_bench;
mod reduction_bench;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== rust-cuda Performance Benchmarks ===\n");

    // Initialize CUDA
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Get device properties
    let device = Device::get_device(0)?;
    let device_name = device.name()?;
    let cc_major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor)?;
    let cc_minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;

    println!("Device: {}", device_name);
    println!("CUDA Compute Capability: {}.{}\n", cc_major, cc_minor);

    // Run SAXPY benchmark
    println!("--- SAXPY Benchmark ---");
    println!("Single-precision A*X + Y (memory bandwidth limited)\n");
    let saxpy_sizes = vec![1_000_000, 10_000_000, 100_000_000];

    for n in saxpy_sizes {
        let bench = saxpy_bench::SaxpyBenchmark::new(n);
        let time_ms = bench.run_rust_cuda(&module, &stream, 100)?;
        let bandwidth = bench.calculate_bandwidth_gbps(time_ms);
        let gflops = bench.calculate_gflops(time_ms);

        println!("  N = {:>12}: {:>8.3} ms, {:>6.1} GB/s, {:>6.1} GFLOPS",
                 n, time_ms, bandwidth, gflops);
    }

    // Run GEMM benchmark
    println!("\n--- GEMM Benchmark ---");
    println!("Matrix multiplication C = A * B (compute intensive)\n");
    let gemm_sizes = vec![(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)];

    for (m, n, k) in gemm_sizes {
        let bench = gemm_bench::GemmBenchmark::new(m, n, k);

        // Naive version
        let time_naive = bench.run_rust_cuda_naive(&module, &stream, 10)?;
        let gflops_naive = bench.calculate_gflops(time_naive);

        // Tiled version
        let time_tiled = bench.run_rust_cuda_tiled(&module, &stream, 10)?;
        let gflops_tiled = bench.calculate_gflops(time_tiled);

        println!("  {}x{}x{}: Naive: {:>8.3} ms ({:>7.1} GFLOPS), Tiled: {:>8.3} ms ({:>7.1} GFLOPS)",
                 m, n, k, time_naive, gflops_naive, time_tiled, gflops_tiled);
    }

    // Run Reduction benchmark
    println!("\n--- Reduction Benchmark ---");
    println!("Sum reduction (memory bandwidth + synchronization)\n");
    let reduction_sizes = vec![1_000_000, 10_000_000, 100_000_000];

    for n in reduction_sizes {
        let bench = reduction_bench::ReductionBenchmark::new(n);
        let time_ms = bench.run_rust_cuda(&module, &stream, 100)?;
        let bandwidth = bench.calculate_bandwidth_gbps(time_ms);

        println!("  N = {:>12}: {:>8.3} ms, {:>6.1} GB/s",
                 n, time_ms, bandwidth);
    }

    println!("\nBenchmarks complete!");
    println!("\nTo compare with native CUDA:");
    println!("  cd native && make && ./saxpy 10000000");
    Ok(())
}
