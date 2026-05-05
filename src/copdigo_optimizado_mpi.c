#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
    Optimización MPI:
    MPI = Message Passing Interface.

    En esta versión se divide la matriz A por filas.
    Cada proceso calcula una parte de la matriz resultado C.

    Proceso 0:
        - Inicializa A, B y C.
        - Reparte filas de A.
        - Envía B completa a todos.
        - Recoge las partes calculadas de C.

    Resto de procesos:
        - Reciben sus filas de A.
        - Reciben B completa.
        - Calculan sus filas de C.
        - Devuelven el resultado al proceso 0.
*/

void matmul_local(double *A_local, double *B, double *C_local, int filas_locales, int N) {

    for (int i = 0; i < filas_locales; i++) {
        for (int k = 0; k < N; k++) {

            double a = A_local[i * N + k];

            for (int j = 0; j < N; j++) {
                C_local[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

void inicializar_matriz(double *M, int N) {
    for (int i = 0; i < N * N; i++) {
        M[i] = (double)(rand() % 10);
    }
}

void inicializar_cero(double *M, int total) {
    for (int i = 0; i < total; i++) {
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

    int rank, size;
    int N = 500;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (N % size != 0) {
        if (rank == 0) {
            printf("Error: N debe ser divisible entre el número de procesos.\n");
            printf("Ejemplo: N=1000 con 2, 4, 5, 10 procesos.\n");
        }

        MPI_Finalize();
        return 1;
    }

    int filas_locales = N / size;

    double *A = NULL;
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = NULL;

    double *A_local = (double *)malloc(filas_locales * N * sizeof(double));
    double *C_local = (double *)malloc(filas_locales * N * sizeof(double));

    if (B == NULL || A_local == NULL || C_local == NULL) {
        printf("Error reservando memoria en proceso %d\n", rank);
        MPI_Finalize();
        return 1;
    }

    inicializar_cero(C_local, filas_locales * N);

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));

        if (A == NULL || C == NULL) {
            printf("Error reservando memoria en proceso 0\n");
            MPI_Finalize();
            return 1;
        }

        inicializar_matriz(A, N);
        inicializar_matriz(B, N);
        inicializar_cero(C, N * N);
    }

    /*
        MPI_Bcast:
        El proceso 0 envía la matriz B completa a todos los procesos.

        Esto es necesario porque todos los procesos necesitan B para calcular
        sus filas de C.
    */
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*
        MPI_Scatter:
        Divide la matriz A por bloques de filas.

        Cada proceso recibe filas_locales filas de A.
    */
    MPI_Scatter(
        A,
        filas_locales * N,
        MPI_DOUBLE,
        A_local,
        filas_locales * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double inicio = MPI_Wtime();

    matmul_local(A_local, B, C_local, filas_locales, N);

    MPI_Barrier(MPI_COMM_WORLD);
    double fin = MPI_Wtime();

    /*
        MPI_Gather:
        Recoge las partes de C calculadas por cada proceso
        y las junta en la matriz C del proceso 0.
    */
    MPI_Gather(
        C_local,
        filas_locales * N,
        MPI_DOUBLE,
        C,
        filas_locales * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        printf("Tamaño matriz: %d x %d\n", N, N);
        printf("Procesos MPI: %d\n", size);
        printf("Tiempo MPI: %f segundos\n", fin - inicio);
        printf("Checksum: %f\n", checksum(C, N));
    }

    free(B);
    free(A_local);
    free(C_local);

    if (rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();

    return 0;
}