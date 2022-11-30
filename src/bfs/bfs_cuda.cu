#include <iostream>
#include <fstream>
#include <cfloat>
#include <cstdlib>
#include <time.h>
#include <ctime>
#include <cuda.h>
#include <stdio.h>

#define BLOCK_NUM 8
#define BLOCK_SIZE 500

bool **parse_input(const std::string &filename, int &sample_size, int &graph_size)
{
  std::ifstream input;
  input.open(filename.c_str());
  if (!input.is_open())
  {
    std::cout << "Error: cannot open file at \"" << filename << "\"" << std::endl;
    exit(1);
  }
  if (!(input >> sample_size))
  {
    std::cout << "Error: cannot read sample size" << std::endl;
    exit(1);
  }

  bool **graph_list = (bool **)malloc(sample_size * sizeof(bool *));
  for (int i = 0; i < sample_size; i++)
  {
    int new_graph_size;
    if (!(input >> new_graph_size))
    {
      std::cout << "Error: cannot read graph size" << std::endl;
      exit(1);
    }
    if (graph_size != 0 && new_graph_size != graph_size)
    {
      std::cout << "Error: graph size mismatch" << std::endl;
      exit(1);
    }
    graph_size = new_graph_size;
    graph_list[i] = (bool *)malloc(graph_size * graph_size * sizeof(bool));
    for (int j = 0; j < graph_size * graph_size; j++)
    {
      graph_list[i][j] = false;
    }
    for (int j = 0; j < graph_size; j++)
    {
      int list_size;
      if (!(input >> list_size))
      {
        std::cout << "Error: cannot read list size" << std::endl;
        exit(1);
      }
      for (int k = 0; k < list_size; k++)
      {
        int neighbor;
        if (!(input >> neighbor))
        {
          std::cout << "Error: cannot read neighbor" << std::endl;
          exit(1);
        }
        graph_list[i][j * graph_size + neighbor] = true;
      }
    }
  }

  input.close();
  return graph_list;
}

__global__ void bfs_kernel(bool *graph, bool *visited, bool *explored, int *frontier, int *next_frontier_size, int *next_frontier, int frontier_size, int graph_size, int stride)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // printf("tid: %d, stride: %d, frontier_size: %d\n", tid, stride, frontier_size);
  for (int i = tid * stride; i < (tid + 1) * stride && i < frontier_size; i++)
  {
    int node = frontier[i];
    visited[node] = true;
    for (int j = 0; j < graph_size; j++)
    {
      if (graph[node * graph_size + j] && !explored[j])
      {
        explored[j] = true;
        int index = atomicAdd(&next_frontier_size[0], 1);
        // printf("index: %d, next_frontier_size: %d, j: %d\n", index, *next_frontier_size, j);
        next_frontier[0] = j;
      }
    }
  }
}

void bfs_graph(bool *graph, int graph_size)
{
  bool *visited = (bool *)malloc(graph_size * sizeof(bool));
  for (int i = 0; i < graph_size; i++)
  {
    visited[i] = false;
  }
  bool *explored = (bool *)malloc(graph_size * sizeof(bool));
  for (int i = 0; i < graph_size; i++)
  {
    explored[i] = false;
  }
  explored[0] = true;
  int *frontier_size = (int *)malloc(sizeof(int));
  *frontier_size = 1;
  int *frontier = (int *)malloc(graph_size * sizeof(int));
  frontier[0] = 0;

  bool *d_graph, *d_visited, *d_explored;
  cudaMalloc((void **)&d_graph, graph_size * graph_size * sizeof(bool));
  cudaMalloc((void **)&d_visited, graph_size * sizeof(bool));
  cudaMalloc((void **)&d_explored, graph_size * sizeof(bool));
  int *d_frontier, *d_next_frontier_size, *d_next_frontier;
  cudaMalloc((void **)&d_frontier, graph_size * sizeof(int));
  cudaMalloc((void **)&d_next_frontier_size, sizeof(int));
  cudaMalloc((void **)&d_next_frontier, graph_size * sizeof(int));

  cudaMemcpy(d_graph, graph, graph_size * graph_size * sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(d_visited, visited, graph_size * sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(d_explored, explored, graph_size * sizeof(bool), cudaMemcpyHostToDevice);

  while (*frontier_size > 0)
  {
    cudaMemcpy(d_frontier, frontier, (*frontier_size) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_next_frontier_size, 0, sizeof(int));

    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(BLOCK_NUM, 1, 1);

    int stride = ceil((double)(*frontier_size) / (BLOCK_NUM * BLOCK_SIZE));
    bfs_kernel<<<dimGrid, dimBlock>>>(d_graph, d_visited, d_explored, d_frontier, d_next_frontier_size, d_next_frontier, *frontier_size, graph_size, stride);
    cudaDeviceSynchronize();

    cudaMemcpy(frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(frontier, d_next_frontier, (*frontier_size) * sizeof(int), cudaMemcpyDeviceToHost);
  }

  cudaFree(d_visited);
  cudaFree(d_explored);
  cudaFree(d_frontier);
  cudaFree(d_next_frontier_size);
  cudaFree(d_next_frontier);

  free(visited);
  free(explored);
  free(frontier);
}

int main(int argc, char *argv[])
{
  if (argc < 2 || argc > 3)
  {
    std::cout << "Usage: " << argv[0] << " <input file> [iterations=1]" << std::endl;
    exit(1);
  }
  std::string filename = argv[1];
  int iterations = 1;
  if (argc == 3)
  {
    iterations = atoi(argv[2]);
  }
  int sample_size = 0;
  int graph_size = 0;
  bool **graph_list = parse_input(filename, sample_size, graph_size);
  std::cout << "sample_size: " << sample_size << std::endl;
  std::cout << "graph_size: " << graph_size << std::endl;

  struct timespec start_time, end_time;
  clock_gettime(CLOCK_REALTIME, &start_time);

  for (int i = 0; i < iterations; i++)
  {
    for (int j = 0; j < sample_size; j++)
    {
      bfs_graph(graph_list[j], graph_size);
      std::cout << "- sample " << j + 1 << "/" << sample_size << " done." << std::endl;
    }
    std::cout << "iteration " << i + 1 << "/" << iterations << " finished." << std::endl;
  }

  clock_gettime(CLOCK_REALTIME, &end_time);

  printf("Total time taken by the GPU part = %lf\n", (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_nsec - start_time.tv_nsec) / 1000000000);

  return 0;
}
