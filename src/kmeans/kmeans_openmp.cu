#include <iostream>
#include <omp.h>

int main() {
    float A1[100], B1[100];
    for (int i = 0; i < 100; i++) {
        A1[i] = 0;
        B1[i] = i;
    }
    float time_start = omp_get_wtime();
#pragma omp target map(tofrom: A1[0:100]) map(to: B1[0:100])
    {
#pragma omp parallel for
        for (int i = 0; i < 100; i++) {
            A1[i] = B1[i];
        }
    }
    float time_end = omp_get_wtime();
    std::cout << "GPU Time: " << time_end - time_start << std::endl;
    float sum1 = 0;
    for (int i = 0; i < 100; i++) {
        sum1 += A1[i];
    }
    std::cout << "GPU Sum: " << sum1 << std::endl;

    float A2[100], B2[100];
    for (int i = 0; i < 100; i++) {
        A2[i] = 0;
        B2[i] = i;
    }
    time_start = omp_get_wtime();
    for (int i = 0; i < 100; i++) {
        A2[i] = B2[i];
    }
    time_end = omp_get_wtime();
    std::cout << "CPU Time: " << time_end - time_start << std::endl;
    float sum2 = 0;
    for (int i = 0; i < 100; i++) {
        sum2 += A2[i];
    }

    return 0;
}
