GPP_BIN=gcc
GPP_STD=c++14
NVCPP_BIN=nvc++

dir_guard=@mkdir -p bin

.PHONY: clean

all: kmeans_cuda kmeans_openmp

kmeans_cuda: src/kmeans/kmeans_cuda.cu
	$(dir_guard)
	$(NVCPP_BIN) -o bin/kmeans_cuda src/kmeans/kmeans_cuda.cu -std=$(GPP_STD)

kmeans_openmp: src/kmeans/kmeans_openmp.cpp
	$(dir_guard)
	$(GPP_BIN) -o bin/kmeans_openmp src/kmeans/kmeans_openmp.cpp -fopenmp -foffload=nvptx-none -std=$(GPP_STD)

