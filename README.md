# nyu-gpu-final-project

Comparing CUDA to OpenMP on GPUs

## Getting Started

### Running on CIMS GPU machines

We experimented on `cuda5.cims.nyu.edu`.

1. Load the CUDA module:

```bash
module load cuda
```

2. Download GCC with OpenMP GPU-offloading support:

```bash
mkdir -p /tmp/$(whoami)
wget -O /tmp/$(whoami)/gcc.zip <PLACEHOLDER_URL> && unzip /tmp/$(whoami)/gcc.zip -d /tmp/$(whoami) && rm -f /tmp/$(whoami)/gcc.zip
```

> **Or alternatively, you can build GCC from source with OpenMP GPU-offload support**
>
> ```bash
> wget -O install.sh https://gist.githubusercontent.com/dizys/8dedbe94439b91d759b6c1e6e316d542/raw/3ddbd8def8cc5bc7ce42549317820df16daf9e96/build_gcc_with_offload.sh && sh install.sh && rm -f install.sh
> ```
>
> This will build GCC 11 from source and take roughly 30 minutes. GCC will be installed to `/tmp/<NET_ID>/gcc` temporarily.

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
