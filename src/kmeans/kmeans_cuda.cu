#include <iostream>
#include <fstream>
#include <cfloat>
#include <cstdlib>
#include <time.h>
#include <cuda.h>
#include <stdio.h>

#define BLOCK_NUM 8
#define BLOCK_SIZE 500

#define K 10
#define DIM 3

float *parse_input(const std::string &filename, long unsigned &vector_size)
{
    std::ifstream input;
    input.open(filename.c_str());
    if (!input.is_open())
    {
        std::cout << "Error: cannot open file at \"" << filename << "\"" << std::endl;
        exit(1);
    }
    if (!(input >> vector_size))
    {
        std::cout << "Error: cannot read vector size" << std::endl;
        exit(1);
    }

    float *vector = (float *)malloc(vector_size * DIM * sizeof(float));
    for (long unsigned i = 0; i < vector_size * DIM; i++)
    {
        if (!(input >> vector[i]))
        {
            std::cout << "Error: cannot read vector element" << std::endl;
            exit(1);
        }
    }

    input.close();
    return vector;
}

void pick_random_centroids(float *centroids, float *vector, long unsigned vector_size)
{
    for (int i = 0; i < K; i++)
    {
        int centroid_index = rand() % vector_size;
        for (int j = 0; j < DIM; j++)
        {
            centroids[i * DIM + j] = vector[centroid_index * DIM + j];
        }
    }
}

// kernel: reassigns each vector to the closest centroid + computes the new centroids
__global__ void kernel(unsigned vector_size, unsigned vector_stride, float *vectors, float *centroids, unsigned *clusters, unsigned *cluster_sizes, bool *changed)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 0)
    {
        changed[0] = false;

        for (int i = 0; i < K; i++)
        {
            cluster_sizes[i] = 0;
        }

        printf("printing from kernel\n");
    }

    __syncthreads();

    for (unsigned j = i * vector_stride; j < (i + 1) * vector_stride && j < vector_size; j++)
    {
        float min_dist = FLT_MAX;
        unsigned min_centroid = 0;
        for (unsigned k = 0; k < K; k++)
        {
            float dist = 0;
            for (unsigned l = 0; l < DIM; l++)
            {
                float diff = vectors[j * DIM + l] - centroids[k * DIM + l];
                dist += diff * diff;
            }
            if (dist < min_dist)
            {
                min_dist = dist;
                min_centroid = k;
            }
        }

        if (clusters[j] != min_centroid)
        {
            clusters[j] = min_centroid;
            changed[0] = true;
        }
        atomicAdd(&cluster_sizes[min_centroid], 1);
    }

    __syncthreads();

    if (i == 0)
    {
        for (unsigned j = 0; j < K; j++)
        {
            for (unsigned k = 0; k < DIM; k++)
            {
                centroids[j * DIM + k] = 0;
            }
        }
    }

    __syncthreads();

    for (unsigned j = i * vector_stride; j < (i + 1) * vector_stride && j < vector_size; j++)
    {
        unsigned cluster = clusters[j];
        for (unsigned k = 0; k < DIM; k++)
        {
            atomicAdd(&centroids[cluster * DIM + k], vectors[j * DIM + k]);
        }
    }

    __syncthreads();

    if (i == 0)
    {
        for (unsigned j = 0; j < K; j++)
        {
            for (unsigned k = 0; k < DIM; k++)
            {
                if (cluster_sizes[j] > 0)
                {
                    centroids[j * DIM + k] /= cluster_sizes[j];
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "usage: " << argv[0] << " filename" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    long unsigned vector_size = 0;
    float *vectors = parse_input(filename, vector_size);
    float *centroids = (float *)malloc(K * DIM * sizeof(float));
    unsigned *clusters = (unsigned *)malloc(vector_size * sizeof(unsigned));
    for (unsigned i = 0; i < vector_size; i++)
    {
        clusters[i] = 1;
    }
    pick_random_centroids(centroids, vectors, vector_size);
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            std::cout << centroids[i * DIM + j] << " ";
        }
        std::cout << std::endl;
    }

    struct timespec start_time, end_time;

    clock_gettime(CLOCK_REALTIME, &start_time);

    float *d_vectors, *d_centroids;
    unsigned *d_clusters, *d_cluster_sizes;

    cudaMalloc((void **)&d_vectors, vector_size * DIM * sizeof(float));
    cudaMalloc((void **)&d_centroids, K * DIM * sizeof(float));
    cudaMalloc((void **)&d_clusters, vector_size * sizeof(unsigned));
    cudaMalloc((void **)&d_cluster_sizes, K * sizeof(unsigned));

    dim3 grid_size(BLOCK_NUM, 1, 1);
    dim3 block_size(BLOCK_SIZE, 1, 1);
    unsigned vector_stride = ceil(vector_size / (float)(grid_size.x * block_size.x));

    cudaMemcpy(d_vectors, vectors, vector_size * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, K * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clusters, clusters, vector_size * sizeof(unsigned), cudaMemcpyHostToDevice);

    bool *changed = (bool *)malloc(sizeof(bool));
    changed[0] = true;
    bool *d_changed;
    cudaMalloc((void **)&d_changed, sizeof(bool));
    int iteration = 0;
    std::cout << "stride: " << vector_stride << std::endl;
    while (changed[0] && iteration < 100)
    {
        kernel<<<grid_size, block_size>>>(vector_size, vector_stride, d_vectors, d_centroids, d_clusters, d_cluster_sizes, d_changed);
        cudaMemcpy(changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(clusters, d_clusters, vector_size * sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(centroids, d_centroids, K * DIM * sizeof(float), cudaMemcpyDeviceToHost);
        iteration++;
        std::cout << "iteration " << iteration << ": " << (changed[0] ? "changed" : "converged") << std::endl;
        for (unsigned i = 0; i < K; i++)
        {
            std::cout << "cluster " << i << ": ";
            unsigned size = 0;
            for (unsigned j = 0; j < vector_size; j++)
            {
                if (clusters[j] == i)
                {
                    size++;
                }
            }
            std::cout << size << std::endl;
        }
        std::cout << "centroids: " << std::endl;
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < DIM; j++)
            {
                std::cout << centroids[i * DIM + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    cudaMemcpy(clusters, d_clusters, vector_size * sizeof(unsigned), cudaMemcpyDeviceToHost);

    cudaFree(d_vectors);
    cudaFree(d_centroids);
    cudaFree(d_clusters);
    cudaFree(d_cluster_sizes);
    cudaFree(d_changed);

    clock_gettime(CLOCK_REALTIME, &end_time);

    printf("Total time taken by the GPU part = %lf\n", (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_nsec - start_time.tv_nsec) / 1000000000);

    free(changed);
    free(vectors);
    free(centroids);
    free(clusters);

    return 0;
}
