#include "parse_args.h"
#include <string.h>
#include <iostream>

void print_help(const std::string &executable, const std::string &description) {
    std::cout << "usage: " << executable << " t k filename labels_file" << std::endl << std::endl;
    std::cout << description << std::endl << std::endl;
    std::cout << "positional arguments:" << std::endl;
    std::cout << "  t            the number of threads, 0 < t <= " << MAX_THREAD_NUM << std::endl;
    std::cout << "  k            the number of clusters, 1 < k <= " << MAX_K << std::endl;
    std::cout << "  filename     the name of the input file that contains data vectors" << std::endl;
    std::cout << "  labels_file  the name of the file that contains data labels" << std::endl;
}

ParsedArgs parse_args(int argc, char **argv, const std::string &description) {
    std::string executable = argv[0];
    if (argc > 1 && strcmp(argv[1], "-h") == 0) {
        print_help(executable, description);
        exit(0);
    } else if (argc != 5) {
        std::cout << "Error: invalid arguments" << std::endl << std::endl;
        print_help(executable, description);
        exit(1);
    }
    std::string thread_count_str = argv[1];
    std::string k_str = argv[2];
    std::string input_filename = argv[3];
    std::string labels_filename = argv[4];

    int thread_count = atoi(thread_count_str.c_str());
    if (thread_count <= 0 || thread_count > MAX_THREAD_NUM) {
        std::cout << "Error: invalid thread count \"" << thread_count_str << "\"" << std::endl << std::endl;
        print_help(executable, description);
        exit(1);
    }

    int k = atoi(k_str.c_str());
    if (k <= 0 || k > MAX_K) {
        std::cout << "Error: invalid k \"" << k_str << "\"" << std::endl << std::endl;
        print_help(executable, description);
        exit(1);
    }

    if (input_filename.empty()) {
        std::cout << "Error: input filename cannot be empty" << std::endl << std::endl;
        print_help(executable, description);
        exit(1);
    }

    if (labels_filename.empty()) {
        std::cout << "Error: labels filename cannot be empty" << std::endl << std::endl;
        print_help(executable, description);
        exit(1);
    }

    ParsedArgs parsed_args;
    parsed_args.thread_count = thread_count;
    parsed_args.k = k;
    parsed_args.input_filename = input_filename;
    parsed_args.labels_filename = labels_filename;
    return parsed_args;
}
