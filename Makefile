GPP_BIN=gcc
GPP_STD=c++11
NVCC_BIN=nvcc

dir_guard=@mkdir -p bin

.PHONY: clean

all: kmeans_seq kmeans_cuda kmeans_openmp

kmeans_seq: src/kmeans/kmeans_seq.cpp
	$(dir_guard)
	$(GPP_BIN) -std=$(GPP_STD) -o bin/kmeans_seq src/kmeans/kmeans_seq.cpp

kmeans_cuda: src/kmeans/kmeans_cuda.cu
	$(dir_guard)
	$(NVCC_BIN) -std=$(GPP_STD) -o bin/kmeans_cuda src/kmeans/kmeans_cuda.cu

kmeans_openmp: src/kmeans/kmeans_openmp.cpp
	$(dir_guard)
	$(GPP_BIN) -std=$(GPP_STD) -o bin/kmeans_openmp -fopenmp -foffload=nvptx-none src/kmeans/kmeans_openmp.cpp

bfs_seq: src/bfs/bfs_seq.cpp
	$(dir_guard)
	$(GPP_BIN) -std=$(GPP_STD) -o bin/bfs_seq src/bfs/bfs_seq.cpp

clean:
	rm -rf bin
