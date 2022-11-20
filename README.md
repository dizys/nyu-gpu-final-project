# nyu-gpu-final-project

Comparing CUDA to OpenMP on GPUs

## Getting Started

### Switch Modules on CIMS machines

```bash
module load cmake-3
module load gcc-11.2
module load cuda-11.4
```

### Build

```bash
cmake -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --config Release
```