# nyu-gpu-final-project

Comparing CUDA to OpenMP on GPUs

## Getting Started

### Running on CIMS GPU machines

We experimented on `cuda5.cims.nyu.edu`.

1. Load the CUDA module:

```bash
module load cuda-10.1
```

2. Build GCC from source with OpenMP GPU-offload support:

```bash
wget -O install.sh https://gist.githubusercontent.com/dizys/8dedbe94439b91d759b6c1e6e316d542/raw/88fd90611513e4caf89cdfb41ae6734ca6e1a2ea/build_gcc_with_offload.sh && sh install.sh && rm -f install.sh
```

This will build GCC 11 from source and install it to `/tmp/<NET_ID>/gcc` temporarily.

### Build

Before compilation, environment variable `LD_LIBRARY_PATH` must be set to the path of the GCC installation.

```bash
export MY_GCC_PATH=/tmp/$(whoami)/gcc
export LD_LIBRARY_PATH=$MY_GCC_PATH/lib64:${LD_LIBRARY_PATH}
```

Then, build the project:

```bash
make GPP_BIN=$MY_GCC_PATH/bin/g++
```
