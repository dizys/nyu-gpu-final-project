GPP_BIN=gcc
GPP_STD=c++11
NVCC_BIN=nvcc

dir_guard=@mkdir -p bin

.PHONY: clean

all: kmeans_seq kmeans_cuda kmeans_openmp bfs_seq bfs_cuda bfs_openmp

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

bfs_openmp: src/bfs/bfs_openmp.cpp
	$(dir_guard)
	$(GPP_BIN) -std=$(GPP_STD) -o bin/bfs_openmp -fopenmp -foffload=nvptx-none src/bfs/bfs_openmp.cpp

nqueens_seq: src/nqueens/nqueens_seq.cpp
	$(dir_guard)
	$(MY_GCC_PATH)/bin/g++ -std=c++11 -o bin/nqueens_seq -fopenmp -foffload=nvptx-none src/nqueens/nqueens_seq.cpp

nqueens_openmp: src/nqueens/nqueens_openmp.cpp
	$(dir_guard)
	$(MY_GCC_PATH)/bin/g++ -std=c++11 -o bin/nqueens_openmp -fopenmp -foffload=nvptx-none src/nqueens/nqueens_openmp.cpp

nqueens_cuda: src/nqueens/nqueens_cuda.cu
	$(dir_guard)
	$(NVCC_BIN) -std=$(GPP_STD) -o bin/nqueens_cuda src/nqueens/nqueens_cuda.cu

warshall_seq: src/warshall/warshall_seq.cpp
	$(dir_guard)
	$(MY_GCC_PATH)/bin/g++ -std=c++11 -o bin/warshall_seq -fopenmp -foffload=nvptx-none src/warshall/warshall_seq.cpp

warshall_openmp: src/warshall/warshall_openmp.cpp
	$(dir_guard)
	$(MY_GCC_PATH)/bin/g++ -std=c++11 -o bin/warshall_openmp -fopenmp -foffload=nvptx-none src/warshall/warshall_openmp.cpp

warshall_cuda: src/warshall/warshall_cuda.cu
	$(dir_guard)
	$(NVCC_BIN) -std=$(GPP_STD) -o bin/warshall_cuda src/warshall/warshall_cuda.cu

generategraph: src/warshall/generategraph.cpp
	$(dir_guard)
	$(MY_GCC_PATH)/bin/g++ -std=c++11 -o bin/generategraph src/warshall/generategraph.cpp

clean:
	rm -rf bin
