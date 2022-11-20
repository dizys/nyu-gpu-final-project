#ifndef KMEANS_PARSE_ARGS_H
#define KMEANS_PARSE_ARGS_H

#include <string>

#define MAX_THREAD_NUM 100
#define MAX_K 100

struct ParsedArgs {
    int thread_count;
    int k;
    std::string input_filename;
    std::string labels_filename;
};

void print_help(const std::string &executable, const std::string &description);

ParsedArgs parse_args(int argc, char **argv, const std::string &description);

#endif
