GPP_BIN=gcc
GPP_STD=c++11
NVCC_BIN=nvcc

dir_guard=@mkdir -p bin

.PHONY: clean

all: kmeans_seq kmeans_cuda kmeans_openmp bfs_seq bfs_cuda

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

bfs_cuda: src/bfs/bfs_cuda.cu
	$(dir_guard)
	$(NVCC_BIN) -std=$(GPP_STD) -o bin/bfs_cuda src/bfs/bfs_cuda.cu

nqueens_seq: src/nqueens/nqueens_seq.cpp
	$(dir_guard)
	$(MY_GCC_PATH)/bin/g++ -std=c++11 -o bin/nqueens_seq -fopenmp -foffload=nvptx-none src/nqueens/nqueens_seq.cpp

nqueens_openmp: src/nqueens/nqueens_openmp.cpp
	$(dir_guard)
	$(MY_GCC_PATH)/bin/g++ -std=c++11 -o bin/nqueens_openmp -fopenmp -foffload=nvptx-none src/nqueens/nqueens_openmp.cpp

nqueens_cuda: src/nqueens/nqueens_cuda.cu
	$(dir_guard)
	$(NVCC_BIN) -std=$(GPP_STD) -o bin/nqueens_cuda src/nqueens/nqueens_cuda.cu

clean:
	rm -rf bin
