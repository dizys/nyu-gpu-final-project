#include <iostream>
#include <omp.h>

#define N 1000000000

int main() {
    double *A1, *B1;
    A1 = (double *) malloc(N * sizeof(double));
    B1 = (double *) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        A1[i] = 0;
        B1[i] = i;
    }
    float time_start = omp_get_wtime();
#pragma omp target map(tofrom: A1[0:N]) map(to: B1[0:N])
    {
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            A1[i] = B1[i];
        }
    }
    float time_end = omp_get_wtime();
    std::cout << "GPU Time: " << time_end - time_start << std::endl;
    double sum1 = 0;
    for (int i = 0; i < N; i++) {
        sum1 += A1[i];
    }
    std::cout << "GPU Sum: " << sum1 << std::endl;

    double *A2, *B2;
    A2 = (double *) malloc(N * sizeof(double));
    B2 = (double *) malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        A2[i] = 0;
        B2[i] = i;
    }
    time_start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        A2[i] = B2[i];
    }
    time_end = omp_get_wtime();
    std::cout << "CPU Time: " << time_end - time_start << std::endl;
    double sum2 = 0;
    for (int i = 0; i < N; i++) {
        sum2 += A2[i];
    }
    std::cout << "CPU Sum: " << sum2 << std::endl;

    free(A1);
    free(B1);
    free(A2);
    free(B2);

    return 0;
}
