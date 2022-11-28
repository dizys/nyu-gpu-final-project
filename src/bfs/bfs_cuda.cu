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

bool **parse_input(const std::string &filename, long unsigned &sample_size, long unsigned &graph_size)
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
  for (long unsigned i = 0; i < sample_size; i++)
  {
    long unsigned new_graph_size;
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
    for (long unsigned j = 0; j < graph_size * graph_size; j++)
    {
      graph_list[i][j] = false;
    }
    for (long unsigned j = 0; j < graph_size; j++)
    {
      long unsigned list_size;
      if (!(input >> list_size))
      {
        std::cout << "Error: cannot read list size" << std::endl;
        exit(1);
      }
      for (long unsigned k = 0; k < list_size; k++)
      {
        long unsigned neighbor;
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

__global__ void bfs_kernel(bool *graph, bool *visited, bool *explored, long unsigned *frontier, long unsigned *next_frontier_size, long unsigned *next_frontier, long unsigned frontier_size, long unsigned graph_size, long unsigned stride)
{
}

void bfs_graph(bool *graph, long unsigned graph_size)
{
  bool *visited = (bool *)malloc(graph_size * sizeof(bool));
  for (long unsigned i = 0; i < graph_size; i++)
  {
    visited[i] = false;
  }
  bool *explored = (bool *)malloc(graph_size * sizeof(bool));
  for (long unsigned i = 0; i < graph_size; i++)
  {
    explored[i] = false;
  }
  explored[0] = true;
  long unsigned frontier_size = 1;
  long unsigned *frontier = (long unsigned *)malloc(graph_size * sizeof(long unsigned));
  frontier[0] = 0;

  bool *d_graph = cudaMalloc(graph_size * graph_size * sizeof(bool));
  bool *d_visited = cudaMalloc(graph_size * sizeof(bool));
  bool *d_explored = cudaMalloc(graph_size * sizeof(bool));
  long unsigned *d_frontier = cudaMalloc(graph_size * sizeof(long unsigned));
  long unsigned *d_next_frontier_size = cudaMalloc(sizeof(long unsigned));
  long unsigned *d_next_frontier = cudaMalloc(graph_size * sizeof(long unsigned));

  cudaMemcpy(d_graph, graph, graph_size * graph_size * sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(d_visited, visited, graph_size * sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(d_explored, explored, graph_size * sizeof(bool), cudaMemcpyHostToDevice);

  while (frontier_size > 0)
  {
    cudaMemcpy(d_frontier, frontier, frontier_size * sizeof(long unsigned), cudaMemcpyHostToDevice);
    cudaMemset(d_next_frontier_size, 0, sizeof(long unsigned));

    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(BLOCK_NUM, 1, 1);

    unsigned long stride = ceil((double)frontier_size / (BLOCK_NUM * BLOCK_SIZE));
    bfs_kernel<<<dimGrid, dimBlock>>>(d_graph, d_visited, d_explored, d_frontier, d_next_frontier_size, d_next_frontier, frontier_size, graph_size, stride);

    cudaMemcpy(frontier_size, d_next_frontier_size, sizeof(long unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(frontier, d_next_frontier, frontier_size * sizeof(long unsigned), cudaMemcpyDeviceToHost);
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
  long unsigned iterations = 1;
  if (argc == 3)
  {
    iterations = atoi(argv[2]);
  }
  long unsigned sample_size = 0;
  long unsigned graph_size = 0;
  bool **graph_list = parse_input(filename, sample_size, graph_size);
  std::cout << "sample_size: " << sample_size << std::endl;
  std::cout << "graph_size: " << graph_size << std::endl;

  struct timespec start_time, end_time;
  clock_gettime(CLOCK_REALTIME, &start_time);

  for (long unsigned i = 0; i < iterations; i++)
  {
    for (long unsigned j = 0; j < sample_size; j++)
    {
      bfs_graph(graph_list[j], graph_size);
      std::cout << "- sample " << j << "/" << sample_size << " done." << std::endl;
    }
    std::cout << "iteration " << i << "/" << iterations << " finished." << std::endl;
  }

  clock_gettime(CLOCK_REALTIME, &end_time);

  printf("Total time taken by the GPU part = %lf\n", (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_nsec - start_time.tv_nsec) / 1000000000);

  return 0;
}
