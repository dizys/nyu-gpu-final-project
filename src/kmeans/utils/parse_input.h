#ifndef KMEANS_PARSE_INPUT_H
#define KMEANS_PARSE_INPUT_H

#include <iostream>
#include <vector>
#include <string>

std::vector<std::vector<double>> *parse_input(const std::string &filename);

std::vector<int> *parse_labels_input(const std::string &filename);

#endif
