# nyu-gpu-final-project

Comparing CUDA to OpenMP on GPUs with K-Means, BFS, Floyd-Warshall, and N-Queens algorithms.

## Getting Started

### Setup CIMS GPU machines

We experimented on `cuda3.cims.nyu.edu`.

1. Load the CUDA module:

```bash
module load cuda-10.1
```

2. Download GCC with OpenMP GPU-offloading support:

```bash
mkdir -p /tmp/$(whoami)
wget -O /tmp/$(whoami)/gcc.zip https://github.com/nyu-multicore/cims-gpu/releases/download/gcc/gcc-11.3.1_cims_gpu_offload_22112501.zip && unzip /tmp/$(whoami)/gcc.zip -d /tmp/$(whoami) && rm -f /tmp/$(whoami)/gcc.zip
```

> **Or for step 2, alternatively, you can build GCC from source with OpenMP GPU-offload support**
>
> ```bash
> wget -O build_gcc.sh https://gist.githubusercontent.com/dizys/8dedbe94439b91d759b6c1e6e316d542/raw/3ddbd8def8cc5bc7ce42549317820df16daf9e96/build_gcc_with_offload.sh && sh build_gcc.sh && rm -f build_gcc.sh
> ```
>
> This will take roughly 30 minutes to build GCC 11 from source. And GCC will be installed to `/tmp/<NET_ID>/gcc` temporarily.

### Compile the programs

Before compilation, environment variable `LD_LIBRARY_PATH` must be set to the path of the GCC installation.

```bash
export MY_GCC_PATH=/tmp/$(whoami)/gcc
export LD_LIBRARY_PATH=$MY_GCC_PATH/lib64:${LD_LIBRARY_PATH}
```

Then, build the project:

```bash
make GPP_BIN=$MY_GCC_PATH/bin/g++
```

### Download datasets

```bash
mkdir -p /tmp/$(whoami)/data

# KMeans datasets
wget -O /tmp/$(whoami)/data/kmeans_10000.txt https://media.githubusercontent.com/media/nyu-multicore/k-means/main/data/dataset-10000.txt
wget -O /tmp/$(whoami)/data/kmeans_100000.txt https://media.githubusercontent.com/media/nyu-multicore/k-means/main/data/dataset-100000.txt
wget -O /tmp/$(whoami)/data/kmeans_1000000.txt https://media.githubusercontent.com/media/nyu-multicore/k-means/main/data/dataset-1000000.txt
wget -O /tmp/$(whoami)/data/kmeans_5000000.txt https://media.githubusercontent.com/media/nyu-multicore/k-means/main/data/dataset-5000000.txt
wget -O /tmp/$(whoami)/data/kmeans_10000000.txt https://media.githubusercontent.com/media/nyu-multicore/k-means/main/data/dataset-10000000.txt

# BFS datasets
wget -O /tmp/$(whoami)/data/graph_g1000_s100.txt https://github.com/nyu-multicore/cims-gpu/releases/download/bfs-data/graph_g1000_s100.txt
wget -O /tmp/$(whoami)/data/graph_g2000_s100.txt https://github.com/nyu-multicore/cims-gpu/releases/download/bfs-data/graph_g2000_s100.txt
wget -O /tmp/$(whoami)/data/graph_g4000_s100.txt https://github.com/nyu-multicore/cims-gpu/releases/download/bfs-data/graph_g4000_s100.txt
wget -O /tmp/$(whoami)/data/graph_g8000_s100.txt https://github.com/nyu-multicore/cims-gpu/releases/download/bfs-data/graph_g8000_s100.txt
wget -O /tmp/$(whoami)/data/graph_g16000_s100.txt https://github.com/nyu-multicore/cims-gpu/releases/download/bfs-data/graph_g16000_s100.txt

# Floyd-Warshall datasets
cd bin && ./generategraph <SIZE> && cd .. # generate dataset <INPUT_FILE_SIZE>.txt

# N-Queens datasets: N-Queens programs don't need any extra dataset files to run, it will generate the dataset on the fly
```

## Run the programs

```bash
cd bin

# KMeans
./kmeans_seq /tmp/$(whoami)/data/kmeans_<SIZE>.txt      # run the sequential version
./kmeans_cuda /tmp/$(whoami)/data/kmeans_<SIZE>.txt     # run the CUDA version
./kmeans_openmp /tmp/$(whoami)/data/kmeans_<SIZE>.txt   # run the OpenMP version

# BFS
./bfs_seq /tmp/$(whoami)/data/graph_g<GRAPH_SIZE>_s<SAMPLE_SIZE>.txt      # run the sequential version
./bfs_cuda /tmp/$(whoami)/data/graph_g<GRAPH_SIZE>_s<SAMPLE_SIZE>.txt     # run the CUDA version
./bfs_openmp /tmp/$(whoami)/data/graph_g<GRAPH_SIZE>_s<SAMPLE_SIZE>.txt   # run the OpenMP version

# Floyd-Warshall
./warshall_seq <INPUT_FILE_SIZE>.txt      # run the sequential version
./warshall_cuda <INPUT_FILE_SIZE>.txt     # run the CUDA version
./warshall_openmp <INPUT_FILE_SIZE>.txt   # run the OpenMP version

# N-Queens
./nqueens_seq <SIZE>      # run the sequential version
./nqueens_cuda <SIZE>     # run the CUDA version
./nqueens_openmp <SIZE>   # run the OpenMP version
```

## Experiment Raw Results

We experimented on `cuda3.cims.nyu.edu`. The raw results of how long each program takes to run on each dataset file can be found in the [exp_data.csv](exp_data.csv) file.

Every experiment setting has been run 5 times and should be averaged to get the final result.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
