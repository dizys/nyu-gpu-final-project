#include <iostream>

int main() {
    float A[100], B[100];
    for (int i = 0; i < 100; i++) {
        A[i] = 0;
        B[i] = i;
    }
    float time_start = omp_get_wtime();
#pragma omp target map(tofrom: A[0:100]) map(to: B[0:100])
    {
#pragma omp parallel for
        for (int i = 0; i < 100; i++) {
            A[i] = B[i];
        }
    }
    float time_end = omp_get_wtime();
    std::cout << "GPU Time: " << time_end - time_start << std::endl;

    time_start = omp_get_wtime();
    for (int i = 0; i < 100; i++) {
        A[i] = B[i];
    }
    time_end = omp_get_wtime();
    std::cout << "CPU Time: " << time_end - time_start << std::endl;

    return 0;
}
