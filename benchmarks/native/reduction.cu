// Native CUDA reduction implementation for comparison
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

__global__ void reduction_sum_kernel(const float *input, float *output, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 10000000;
    int iterations = 100;

    printf("Native CUDA Reduction: N = %d\n", n);

    // Allocate host memory
    float *h_input = (float*)malloc(n * sizeof(float));

    // Initialize data
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }

    // Allocate device memory
    float *d_input, *d_output;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, grid_size * sizeof(float));

    // Copy to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) {
        reduction_sum_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
            d_input, d_output, n);
    }
    cudaDeviceSynchronize();

    // Benchmark
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        reduction_sum_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
            d_input, d_output, n);
    }
    cudaDeviceSynchronize();
    double elapsed = get_time() - start;

    double avg_time_ms = (elapsed * 1000.0) / iterations;
    double bandwidth_gbps = (n * sizeof(float)) / (avg_time_ms / 1000.0) / 1e9;

    printf("Time: %.3f ms\n", avg_time_ms);
    printf("Bandwidth: %.1f GB/s\n", bandwidth_gbps);

    // Cleanup
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
