NVCC = nvcc
NVCC_FLAGS = -O3 -use_fast_math -arch=sm_70

all: saxpy gemm reduction

saxpy: saxpy.cu
	$(NVCC) $(NVCC_FLAGS) -o saxpy saxpy.cu

gemm: gemm.cu
	$(NVCC) $(NVCC_FLAGS) -o gemm gemm.cu

reduction: reduction.cu
	$(NVCC) $(NVCC_FLAGS) -o reduction reduction.cu

clean:
	rm -f saxpy gemm reduction

.PHONY: all clean
