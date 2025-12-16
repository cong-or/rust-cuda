// Native CUDA GEMM implementation for comparison
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Tiled GEMM kernel with shared memory
#define TILE_SIZE 16

__global__ void gemm_tiled_kernel(int m, int n, int k,
                                   const float *a, const float *b, float *c) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    int col = blockIdx.y * TILE_SIZE + threadIdx.y;

    float sum = 0.0f;

    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile of A
        int a_col = tile * TILE_SIZE + threadIdx.y;
        if (row < m && a_col < k) {
            tile_a[threadIdx.x][threadIdx.y] = a[row * k + a_col];
        } else {
            tile_a[threadIdx.x][threadIdx.y] = 0.0f;
        }

        // Load tile of B
        int b_row = tile * TILE_SIZE + threadIdx.x;
        if (b_row < k && col < n) {
            tile_b[threadIdx.x][threadIdx.y] = b[b_row * n + col];
        } else {
            tile_b[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_a[threadIdx.x][i] * tile_b[i][threadIdx.y];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    int m = (argc > 1) ? atoi(argv[1]) : 1024;
    int n = (argc > 2) ? atoi(argv[2]) : 1024;
    int k = (argc > 3) ? atoi(argv[3]) : 1024;
    int iterations = 10;

    printf("Native CUDA GEMM: %dx%dx%d\n", m, n, k);

    // Allocate host memory
    float *h_a = (float*)malloc(m * k * sizeof(float));
    float *h_b = (float*)malloc(k * n * sizeof(float));
    float *h_c = (float*)malloc(m * n * sizeof(float));

    // Initialize data
    for (int i = 0; i < m * k; i++) {
        h_a[i] = (float)(i % 100) / 100.0f;
    }
    for (int i = 0; i < k * n; i++) {
        h_b[i] = (float)(i % 100) / 100.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * k * sizeof(float));
    cudaMalloc(&d_b, k * n * sizeof(float));
    cudaMalloc(&d_c, m * n * sizeof(float));

    // Copy to device
    cudaMemcpy(d_a, h_a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((m + TILE_SIZE - 1) / TILE_SIZE,
                  (n + TILE_SIZE - 1) / TILE_SIZE);

    // Warmup
    for (int i = 0; i < 5; i++) {
        gemm_tiled_kernel<<<grid_dim, block_dim>>>(m, n, k, d_a, d_b, d_c);
    }
    cudaDeviceSynchronize();

    // Benchmark
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        gemm_tiled_kernel<<<grid_dim, block_dim>>>(m, n, k, d_a, d_b, d_c);
    }
    cudaDeviceSynchronize();
    double elapsed = get_time() - start;

    double avg_time_ms = (elapsed * 1000.0) / iterations;
    double gflops = (2.0 * m * n * k) / (avg_time_ms / 1000.0) / 1e9;

    printf("Time: %.3f ms\n", avg_time_ms);
    printf("GFLOPS: %.1f\n", gflops);

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
