cmake_minimum_required(VERSION 3.17)
project(nyu_gpu_final_project)

set(CMAKE_CXX_STANDARD 17)

add_executable(kmeans_cuda src/kmeans/kmeans_cuda.cu)
add_executable(kmeans_openmp src/kmeans/kmeans_openmp.cpp)

set_target_properties(kmeans_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(kmeans_openmp PUBLIC OpenMP::OpenMP_CXX)
endif ()
