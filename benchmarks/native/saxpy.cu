// Native CUDA SAXPY implementation for comparison
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

__global__ void saxpy_kernel(int n, float a, const float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 10000000;
    float a = 2.5f;
    int iterations = 100;

    printf("Native CUDA SAXPY: N = %d\n", n);

    // Allocate host memory
    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_y = (float*)malloc(n * sizeof(float));

    // Initialize data (same as rust-cuda)
    for (int i = 0; i < n; i++) {
        h_x[i] = (float)(i % 100) / 100.0f;
        h_y[i] = (float)(i % 50) / 50.0f;
    }

    // Allocate device memory
    float *d_x, *d_y;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    // Copy to device
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // Warmup
    for (int i = 0; i < 10; i++) {
        saxpy_kernel<<<grid_size, block_size>>>(n, a, d_x, d_y);
    }
    cudaDeviceSynchronize();

    // Benchmark
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        saxpy_kernel<<<grid_size, block_size>>>(n, a, d_x, d_y);
    }
    cudaDeviceSynchronize();
    double elapsed = get_time() - start;

    double avg_time_ms = (elapsed * 1000.0) / iterations;
    double bandwidth_gbps = (3.0 * n * sizeof(float)) / (avg_time_ms / 1000.0) / 1e9;
    double gflops = (2.0 * n) / (avg_time_ms / 1000.0) / 1e9;

    printf("Time: %.3f ms\n", avg_time_ms);
    printf("Bandwidth: %.1f GB/s\n", bandwidth_gbps);
    printf("GFLOPS: %.1f\n", gflops);

    // Cleanup
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
