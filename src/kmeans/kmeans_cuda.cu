#include <iostream>
#include <fstream>

#define K 10

double *parse_input(const std::string &filename, long unsigned &vector_size)
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

    double *vector = new double[vector_size];
    for (long unsigned i = 0; i < vector_size; i++)
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

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "usage: " << argv[0] << " filename" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    long unsigned vector_size = 0;
    double *vector = parse_input(filename, vector_size);
    std::cout << "hello world: " << vector_size << std::endl;
    delete (vector);
    return 0;
}
