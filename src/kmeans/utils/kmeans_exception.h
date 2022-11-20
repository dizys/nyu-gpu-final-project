#ifndef KMEANS_KMEANS_EXCEPTION
#define KMEANS_KMEANS_EXCEPTION

#include <iostream>
#include <utility>

class KMeansException : public std::exception {
private:
    std::string msg;
public:
    explicit KMeansException(std::string msg);

    std::string getMessage();
};

#endif
