#include <iostream>
#include <fstream>
#include <cfloat>
#include <cstdlib>
#include <time.h>
#include <ctime>
#include <stdio.h>

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
  long unsigned *queue = (long unsigned *)malloc(graph_size * sizeof(long unsigned));
  long unsigned queue_start = 0;
  long unsigned queue_end = 0;
  queue[queue_end++] = 0;
  explored[0] = true;
  while (queue_start < queue_end)
  {
    long unsigned node = queue[queue_start++];
    for (long unsigned i = 0; i < graph_size; i++)
    {
      if (graph[node * graph_size + i] && !explored[i])
      {
        queue[queue_end++] = i;
        explored[i] = true;
      }
    }
    visited[node] = true;
  }
  free(visited);
  free(explored);
  free(queue);
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

  printf("Total time taken by the CPU part = %lf\n", (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_nsec - start_time.tv_nsec) / 1000000000);

  return 0;
}
