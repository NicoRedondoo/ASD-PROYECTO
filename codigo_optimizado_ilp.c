#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
    Optimización ILP:
    - Mejora el paralelismo a nivel de instrucciones dentro del procesador.
*/

void matmul_ilp(double *A, double *B, double *C, int N) {

    for (int i = 0; i < N; i++) {

        for (int k = 0; k < N; k++) {

            /*
                ILP 1: reutilización de dato.
                Guardamos A[i][k] en una variable local para evitar
                acceder repetidamente a memoria.
            */
            double a = A[i * N + k];

            int j = 0;

            /*
                ILP 2: desenrollado del bucle.
                En vez de hacer una operación por iteración,
                hacemos 4 operaciones independientes.

                Esto permite que el procesador ejecute varias instrucciones
                en paralelo dentro del pipeline.
            */
            for (; j + 3 < N; j += 4) {
                C[i * N + j]     += a * B[k * N + j];
                C[i * N + j + 1] += a * B[k * N + j + 1];
                C[i * N + j + 2] += a * B[k * N + j + 2];
                C[i * N + j + 3] += a * B[k * N + j + 3];
            }

            /*
                Resto del bucle por si N no es múltiplo de 4.
            */
            for (; j < N; j++) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

void inicializar_matriz(double *M, int N) {
    for (int i = 0; i < N * N; i++) {
        M[i] = (double)(rand() % 10);
    }
}

void inicializar_cero(double *M, int N) {
    for (int i = 0; i < N * N; i++) {
        M[i] = 0.0;
    }
}

double checksum(double *M, int N) {
    double suma = 0.0;

    for (int i = 0; i < N * N; i++) {
        suma += M[i];
    }

    return suma;
}   

int main(int argc, char **argv) {

    int N = 500;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        printf("Error reservando memoria\n");
        return 1;
    }

    inicializar_matriz(A, N);
    inicializar_matriz(B, N);
    inicializar_cero(C, N);

    double inicio = omp_get_wtime();

    matmul_ilp(A, B, C, N);

    double fin = omp_get_wtime();

    printf("Tamaño matriz: %d x %d\n", N, N);
    printf("Tiempo ILP: %f segundos\n", fin - inicio);
    printf("Checksum: %f\n", checksum(C, N));

    free(A);
    free(B);
    free(C);

    return 0;
}