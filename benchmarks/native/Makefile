NVCC = nvcc
NVCC_FLAGS = -O3 -use_fast_math -arch=sm_70
CUBLAS_FLAGS = -lcublas

all: saxpy gemm gemm_cublas reduction

saxpy: saxpy.cu
	$(NVCC) $(NVCC_FLAGS) -o saxpy saxpy.cu

gemm: gemm.cu
	$(NVCC) $(NVCC_FLAGS) -o gemm gemm.cu

gemm_cublas: gemm_cublas.cu
	$(NVCC) $(NVCC_FLAGS) $(CUBLAS_FLAGS) -o gemm_cublas gemm_cublas.cu

reduction: reduction.cu
	$(NVCC) $(NVCC_FLAGS) -o reduction reduction.cu

clean:
	rm -f saxpy gemm gemm_cublas reduction

.PHONY: all clean
