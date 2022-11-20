#include "parse_input.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include "string_helper.h"

std::vector<std::vector<double>> *parse_input(const std::string &filename) {
    std::ifstream input;
    input.open(filename);
    if (!input.is_open()) {
        std::cout << "Error: cannot open file at \"" << filename << "\"" << std::endl;
        exit(1);
    }
    std::string line;

    auto *data = new std::vector<std::vector<double>>();

    int line_num = -1;
    long unsigned vector_count = 0;
    long unsigned vector_dim = 0;
    while (std::getline(input, line)) {
        line_num++;

        trim(line);

        if (line_num == 0) {
            try {
                vector_count = std::stoi(line);
            } catch (const std::invalid_argument &ia) {
                std::cout << "Error: first line of input file must be an integer" << std::endl;
                exit(1);
            }
            continue;
        }

        if (line.empty() || data->size() >= vector_count) {
            continue;
        }

        std::vector<double> vector;

        std::istringstream iss(line);
        for (std::string token; iss >> token;) {
            try {
                vector.push_back(std::stod(token));
            } catch (const std::invalid_argument &ia) {
                std::cout << "Error on input line " << line_num + 1 << ": invalid number \"" << token << "\""
                          << std::endl;
                exit(1);
            }
        }

        if (vector_dim == 0) {
            vector_dim = vector.size();
        } else if (vector_dim != vector.size()) {
            std::cout << "Error on input line " << line_num + 1 << ": vector has " << vector.size()
                      << " dimensions, but previous vectors have " << vector_dim << " dimensions" << std::endl;
            exit(1);
        }

        data->push_back(vector);
    }

    return data;
}

std::vector<int> *parse_labels_input(const std::string &filename) {
    std::ifstream input;
    input.open(filename);
    if (!input.is_open()) {
        std::cout << "Error: cannot open file at \"" << filename << "\"" << std::endl;
        exit(1);
    }
    std::string line;

    auto *labels = new std::vector<int>();

    int line_num = -1;
    long unsigned label_count = 0;
    while (std::getline(input, line)) {
        line_num++;

        trim(line);

        if (line_num == 0) {
            try {
                label_count = std::stoi(line);
            } catch (const std::invalid_argument &ia) {
                std::cout << "Error: first line of input file must be an integer" << std::endl;
                exit(1);
            }
            continue;
        }

        if (line.empty() || labels->size() >= label_count) {
            continue;
        }

        try {
            labels->push_back(std::stoi(line));
        } catch (const std::invalid_argument &ia) {
            std::cout << "Error on input line " << line_num + 1 << ": invalid number \"" << line << "\"" << std::endl;
            exit(1);
        }
    }

    return labels;
}
