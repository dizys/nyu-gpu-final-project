# nyu-gpu-final-project

Comparing CUDA to OpenMP on GPUs

## Getting Started

### Setup CIMS GPU machines

We experimented on `cuda5.cims.nyu.edu`.

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

# BFS datasets
wget -O /tmp/$(whoami)/data/graph_g1000_s100.txt https://github.com/nyu-multicore/cims-gpu/releases/download/bfs-data/graph_g1000_s100.txt
wget -O /tmp/$(whoami)/data/graph_g10000_s100.txt https://github.com/nyu-multicore/cims-gpu/releases/download/bfs-data/graph_g10000_s100.txt
```
