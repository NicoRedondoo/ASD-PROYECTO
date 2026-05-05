#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Multiplicación secuencial de matrices
void matmul_seq(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a = A[i*N + k];
            for (int j = 0; j < N; j++) {
                C[i*N + j] += a * B[k*N + j];
            }
        }
    }
}

// Inicializar matrices con valores
void init_matrix(double *M, int N) {
    for (int i = 0; i < N*N; i++) {
        M[i] = (double)(rand() % 10); // valores entre 0 y 9
    }
}

// Poner matriz a 0
void zero_matrix(double *M, int N) {
    for (int i = 0; i < N*N; i++) {
        M[i] = 0.0;
    }
}

int main() {
    int N = 500; // tamaño de la matriz (cámbialo para probar)

    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(N * N * sizeof(double));

    // Inicializar
    init_matrix(A, N);
    init_matrix(B, N);
    zero_matrix(C, N);

    // Medir tiempo
    double t0 = omp_get_wtime();

    matmul_seq(A, B, C, N);

    double t1 = omp_get_wtime();

    printf("Tiempo de ejecución (secuencial): %f segundos\n", t1 - t0);

    // Liberar memoria
    free(A);
    free(B);
    free(C);

    return 0;
}