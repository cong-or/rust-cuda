# rust-cuda Performance Benchmarks

This directory contains benchmarks comparing rust-cuda performance against native CUDA implementations.

## Benchmarks

1. **SAXPY** - Single-precision A*X + Y (measures basic arithmetic throughput)
2. **GEMM** - Matrix multiplication (measures compute-bound performance)
3. **Reduction** - Sum reduction (measures memory bandwidth)

## Quick Setup (Cloud GPU)

For cloud GPU instances (RunPod, Lambda Labs, Vast.ai, etc.):

```bash
# Download and run setup script
curl -sSf https://raw.githubusercontent.com/cong-or/rust-cuda/main/benchmarks/setup_cloud_gpu.sh | bash

# Or if you've already cloned the repo:
cd benchmarks
bash setup_cloud_gpu.sh
```

This will automatically:
- Install all dependencies
- Build LLVM 7.1.0
- Clone and build rust-cuda
- Build benchmarks
- Run tests

## Running Benchmarks

```bash
cd benchmarks
cargo run --release
```

## Results

See [guide/src/benchmarks.md](../guide/src/benchmarks.md) for detailed results and analysis.

## Comparison Methodology

- All benchmarks run 100 iterations with warmup
- Median time reported to reduce variance
- Native CUDA compiled with `-O3 -use_fast_math`
- Measured on: [GPU model, CUDA version]
- Input sizes chosen to saturate GPU (>10ms runtime)
