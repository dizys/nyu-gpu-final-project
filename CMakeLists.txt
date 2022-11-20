cmake_minimum_required(VERSION 3.17)
project(nyu_gpu_final_project CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "-std=c++17")

add_executable(kmeans_cuda src/kmeans/kmeans_cuda.cu)
add_executable(kmeans_openmp src/kmeans/kmeans_openmp.cu)

set_target_properties(kmeans_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(kmeans_openmp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "–mp=gpu ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    target_link_libraries(kmeans_openmp PUBLIC OpenMP::OpenMP_CXX)
endif ()
