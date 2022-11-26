#include <iostream>
#include <fstream>
#include <cfloat>
#include <cstdlib>
#include <time.h>
#include <ctime>
#include <omp.h>
#include <stdio.h>

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

bool assign_clusters(unsigned vector_size, float *vectors, float *centroids, unsigned *clusters, unsigned *cluster_sizes)
{
    bool changed = false;
    for (int i = 0; i < K; i++)
    {
        cluster_sizes[i] = 0;
    }
#pragma omp target map(tofrom                                                                                                                  \
                       : changed) map(tofrom                                                                                                   \
                                      : clusters [0:vector_size]) map(to                                                                       \
                                                                      : vectors [0:vector_size * DIM]) map(to                                  \
                                                                                                           : centroids [0:K * DIM]) map(tofrom \
                                                                                                                                        : cluster_sizes [0:K])
    {
#pragma omp parallel for
        for (unsigned i = 0; i < vector_size; i++)
        {
            float min_distance = FLT_MAX;
            unsigned min_cluster = 0;
            for (unsigned j = 0; j < K; j++)
            {
                float distance = 0;
                for (unsigned k = 0; k < DIM; k++)
                {
                    float diff = vectors[i * DIM + k] - centroids[j * DIM + k];
                    distance += diff * diff;
                }
                if (distance < min_distance)
                {
                    min_distance = distance;
                    min_cluster = j;
                }
            }

            {
                if (clusters[i] != min_cluster)
                {
                    changed = true;
                    clusters[i] = min_cluster;
                }

#pragma omp critical
                {
                    cluster_sizes[min_cluster]++;
                }
            }
        }
    }
    return changed;
}

void compute_centroids(unsigned vector_size, float *vectors, float *centroids, unsigned *clusters, unsigned *cluster_sizes)
{
    for (unsigned i = 0; i < K; i++)
    {
        for (unsigned j = 0; j < DIM; j++)
        {
            centroids[i * DIM + j] = 0;
        }
    }

    for (unsigned i = 0; i < vector_size; i++)
    {
        unsigned cluster = clusters[i];
        for (unsigned j = 0; j < DIM; j++)
        {
            centroids[cluster * DIM + j] += vectors[i * DIM + j];
        }
    }

    for (unsigned i = 0; i < K; i++)
    {
        for (unsigned j = 0; j < DIM; j++)
        {
            centroids[i * DIM + j] /= cluster_sizes[i];
        }
    }
}

int main(int argc, char *argv[])
{
    std::srand(std::time(nullptr));
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
    unsigned *cluster_sizes = (unsigned *)calloc(K, sizeof(unsigned));
    for (unsigned i = 0; i < vector_size; i++)
    {
        clusters[i] = 0;
    }
    pick_random_centroids(centroids, vectors, vector_size);

    struct timespec start_time, end_time;

    clock_gettime(CLOCK_REALTIME, &start_time);

    int iteration = 0;
    bool changed = true;
    while (iteration < 10)
    {
        changed = assign_clusters(vector_size, vectors, centroids, clusters, cluster_sizes);
        compute_centroids(vector_size, vectors, centroids, clusters, cluster_sizes);
        iteration++;
        std::cout << "Iteration #" << iteration << ": " << (changed ? "centroids changed, continuing..." : "converged.") << std::endl;
    }

    clock_gettime(CLOCK_REALTIME, &end_time);

    printf("Total time taken by the GPU part = %lf\n", (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_nsec - start_time.tv_nsec) / 1000000000);

    free(vectors);
    free(centroids);
    free(clusters);
    free(cluster_sizes);

    return 0;
}
