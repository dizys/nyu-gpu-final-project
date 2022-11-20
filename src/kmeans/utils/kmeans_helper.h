#ifndef KMEANS_KMEANS_HELPER_H
#define KMEANS_KMEANS_HELPER_H

#include <vector>
#include <cmath>
#include <iostream>
#include <unordered_set>

static inline double pairwise_distance(std::vector<double> &v1, std::vector<double> &v2) {
    double sum = 0;
    for (int i = 0; i < (int) v1.size(); i++) {
        sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return sqrt(sum);
}

static inline std::vector<std::unordered_set<int>> *get_clusters_by_assignments(std::vector<int> &assignments, int k) {
    auto *clusters = new std::vector<std::unordered_set<int>>(k);
    for (int i = 0; i < (int) assignments.size(); i++) {
        (*clusters)[assignments[i]].insert(i);
    }
    return clusters;
}

static inline long long get_intersection_size(std::unordered_set<int> &s1, std::unordered_set<int> &s2) {
    long long intersection_size = 0;
    for (int it: s1) {
        if (s2.find(it) != s2.end()) {
            intersection_size++;
        }
    }
    return intersection_size;
}


// Calculate normalized mutual information score between two vectors
static inline double compute_assignments_nmi(std::vector<int> &a1, std::vector<int> &a2, int k) {
    if (a1.size() != a2.size()) {
        std::cout << "Error: assignment sizes do not match" << std::endl;
        exit(1);
    }
    auto *clusters1 = get_clusters_by_assignments(a1, k);
    auto *clusters2 = get_clusters_by_assignments(a2, k);
    double num = 0;
    for (auto &set1: *clusters1) {
        for (auto &set2: *clusters2) {
            auto n_i = (long long) set1.size();
            auto n_j = (long long) set2.size();
            long long n_ij = get_intersection_size(set1, set2);
            if (n_ij == 0) {
                continue;
            }
            double log_term = log((double) n_ij * (double) a1.size() / ((double) n_i * (double) n_j));
            double item = (double) n_ij * log_term;
            if (!std::isnan(item)) {
                num += item;
            }
        }
    }
    num *= -2;
    double den = 0;
    for (auto &set1: *clusters1) {
        int n_i = (int) set1.size();
        double item = (double) n_i * log((double) n_i / (double) a1.size());
        if (!std::isnan(item)) {
            den += item;
        }
    }
    for (auto &set2: *clusters2) {
        int n_j = (int) set2.size();
        double item = (double) n_j * log((double) n_j / (double) a2.size());
        if (!std::isnan(item)) {
            den += item;
        }
    }
    delete clusters1;
    delete clusters2;
    return num / den;
}

#endif //KMEANS_KMEANS_HELPER_H
