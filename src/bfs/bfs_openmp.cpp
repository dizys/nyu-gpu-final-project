#include <iostream>
#include <fstream>
#include <cfloat>
#include <cstdlib>
#include <time.h>
#include <ctime>
#include <omp.h>
#include <stdio.h>

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
  int frontier_size = 1;
  int *frontier = (int *)malloc(graph_size * sizeof(int));
  frontier[0] = 0;
  int *next_frontier = (int *)malloc(graph_size * sizeof(int));
  int next_frontier_size = 0;
  while (frontier_size > 0)
  {
#pragma omp target map(to                                              \
                       : graph [0:graph_size * graph_size]) map(tofrom \
                                                                : explored [0:graph_size], visited [0:graph_size], next_frontier [0:graph_size], next_frontier_size)
    {
#pragma omp parallel for
      for (int i = 0; i < frontier_size; i++)
      {
        int node = frontier[i];
        visited[node] = true;
        for (int j = 0; j < graph_size; j++)
        {

          if (graph[node * graph_size + j] && !explored[j])
          {
#pragma omp critical
            {
              next_frontier[next_frontier_size] = j;
              next_frontier_size++;
            }
            explored[j] = true;
          }
        }
      }
    }
    frontier_size = next_frontier_size;
    next_frontier_size = 0;
    int *temp = frontier;
    frontier = next_frontier;
    next_frontier = temp;
  }
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
