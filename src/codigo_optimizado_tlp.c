#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
    Optimización TLP:
    TLP = Thread Level Parallelism.

    En esta versión usamos OpenMP para repartir el cálculo
    de la matriz C entre varios hilos.

    Cada hilo calcula distintas filas de la matriz C.
*/

void matmul_tlp(double *A, double *B, double *C, int N) {

    /*
        OPTIMIZACIÓN TLP:
        #pragma omp parallel for crea varios hilos.

        Cada hilo ejecuta una parte distinta del bucle i.
        Como cada i representa una fila distinta de C,
        no hay conflictos de escritura entre hilos.
    */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {

        for (int k = 0; k < N; k++) {

            /*
                Reutilizamos A[i][k] en variable local.
                Esto no es TLP como tal, pero evita accesos repetidos a memoria.
            */
            double a = A[i * N + k];

            for (int j = 0; j < N; j++) {
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

    matmul_tlp(A, B, C, N);

    double fin = omp_get_wtime();

    printf("Tamaño matriz: %d x %d\n", N, N);
    printf("Hilos usados: %d\n", omp_get_max_threads());
    printf("Tiempo TLP OpenMP: %f segundos\n", fin - inicio);
    printf("Checksum: %f\n", checksum(C, N));

    free(A);
    free(B);
    free(C);

    return 0;
}