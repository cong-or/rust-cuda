// cuBLAS GEMM implementation for comparison
// This represents the optimal GPU performance achievable
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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

    printf("cuBLAS GEMM: %dx%dx%d\n", m, n, k);

    // Allocate host memory
    float *h_a = (float*)malloc(m * k * sizeof(float));
    float *h_b = (float*)malloc(k * n * sizeof(float));
    float *h_c = (float*)malloc(m * n * sizeof(float));

    // Initialize data (same as other benchmarks)
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

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS uses column-major, so we need to transpose the operation
    // C = A * B in row-major is equivalent to C^T = B^T * A^T in column-major
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    d_b, n,
                    d_a, k,
                    &beta,
                    d_c, n);
    }
    cudaDeviceSynchronize();

    // Benchmark
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha,
                    d_b, n,
                    d_a, k,
                    &beta,
                    d_c, n);
    }
    cudaDeviceSynchronize();
    double elapsed = get_time() - start;

    double avg_time_ms = (elapsed * 1000.0) / iterations;
    double gflops = (2.0 * m * n * k) / (avg_time_ms / 1000.0) / 1e9;

    printf("Time: %.3f ms\n", avg_time_ms);
    printf("GFLOPS: %.1f\n", gflops);

    // Cleanup
    cublasDestroy(handle);
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
