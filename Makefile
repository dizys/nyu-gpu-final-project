GPP_BIN=gcc
GPP_STD=c++17
NVCC_BIN=nvcc

.PHONY: clean

kmeans_cuda: src/kmeans/kmeans_cuda.cu
	$(NVCC_BIN) -o bin/kmeans_cuda src/kmeans/kmeans_cuda.cu

kmeans_openmp: src/kmeans/kmeans_openmp.cpp
	$(GPP_BIN) -o bin/kmeans_openmp src/kmeans/kmeans_openmp.cpp -fopenmp -foffload=nvptx-none -std=$(GPP_STD)
	
