#include <iostream>
#include <fstream>

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

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cout << "usage: " << argv[0] << " filename" << std::endl;
    return 1;
  }
  std::string filename = argv[1];
  long unsigned sample_size = 0;
  long unsigned graph_size = 0;
  bool **graph_list = parse_input(filename, sample_size, graph_size);
  std::cout << "Hello World!" << std::endl;
  return 0;
}
