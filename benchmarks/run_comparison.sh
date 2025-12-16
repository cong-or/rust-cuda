#!/bin/bash
# Script to run rust-cuda and native CUDA benchmarks and compare results

set -e

echo "==================================="
echo "rust-cuda vs Native CUDA Benchmarks"
echo "==================================="
echo ""

# Check if native benchmarks are built
if [ ! -f native/saxpy ] || [ ! -f native/gemm ] || [ ! -f native/reduction ]; then
    echo "Building native CUDA benchmarks..."
    cd native
    make
    cd ..
    echo ""
fi

# Run rust-cuda benchmarks
echo "Running rust-cuda benchmarks..."
echo "-----------------------------------"
cargo run --release > rust_cuda_results.txt 2>&1
cat rust_cuda_results.txt
echo ""

# Run native CUDA benchmarks
echo "Running native CUDA benchmarks..."
echo "-----------------------------------"

echo "Native SAXPY:"
./native/saxpy 1000000 > native_saxpy_1m.txt
cat native_saxpy_1m.txt
echo ""

./native/saxpy 10000000 > native_saxpy_10m.txt
cat native_saxpy_10m.txt
echo ""

./native/saxpy 100000000 > native_saxpy_100m.txt
cat native_saxpy_100m.txt
echo ""

echo "Native GEMM:"
./native/gemm 512 512 512 > native_gemm_512.txt
cat native_gemm_512.txt
echo ""

./native/gemm 1024 1024 1024 > native_gemm_1024.txt
cat native_gemm_1024.txt
echo ""

./native/gemm 2048 2048 2048 > native_gemm_2048.txt
cat native_gemm_2048.txt
echo ""

echo "Native Reduction:"
./native/reduction 1000000 > native_reduction_1m.txt
cat native_reduction_1m.txt
echo ""

./native/reduction 10000000 > native_reduction_10m.txt
cat native_reduction_10m.txt
echo ""

./native/reduction 100000000 > native_reduction_100m.txt
cat native_reduction_100m.txt
echo ""

echo "==================================="
echo "Benchmark comparison complete!"
echo "Results saved to:"
echo "  - rust_cuda_results.txt"
echo "  - native_*.txt"
echo "==================================="
