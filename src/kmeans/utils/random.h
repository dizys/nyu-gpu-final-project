#ifndef KMEANS_RANDOM_H
#define KMEANS_RANDOM_H


#include <random>
#include <vector>
#include <string>

static inline double random_double(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

#endif //KMEANS_RANDOM_H
