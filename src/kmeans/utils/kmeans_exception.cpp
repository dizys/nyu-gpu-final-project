#include "kmeans_exception.h"

KMeansException::KMeansException(std::string msg) : msg(std::move(msg)) {
}

std::string KMeansException::getMessage() {
    return msg;
}
