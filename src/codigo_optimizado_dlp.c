#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

/*
    Optimización DLP:
    DLP = Data Level Parallelism.

    En esta versión usamos instrucciones SIMD AVX2.
    Cada registro __m256d puede almacenar 4 valores double.

    Por tanto, en vez de calcular:

        C[i][j]     += A[i][k] * B[k][j]
        C[i][j + 1] += A[i][k] * B[k][j + 1]
        C[i][j + 2] += A[i][k] * B[k][j + 2]
        C[i][j + 3] += A[i][k] * B[k][j + 3]

    lo hacemos en una sola operación vectorial.
*/

void matmul_dlp(double *A, double *B, double *C, int N) {

    for (int i = 0; i < N; i++) {

        for (int k = 0; k < N; k++) {

            /*
                DLP 1:
                Cargamos A[i][k] y lo replicamos 4 veces en un vector AVX.
            */
            __m256d a_vec = _mm256_set1_pd(A[i * N + k]);

            int j = 0;

            /*
                DLP 2:
                Procesamos 4 columnas de golpe.

                loadu_pd  -> carga 4 doubles de memoria
                fmadd_pd  -> hace a*b + c en una sola instrucción
                storeu_pd -> guarda 4 doubles en memoria
            */
            for (; j + 3 < N; j += 4) {

                __m256d b_vec = _mm256_loadu_pd(&B[k * N + j]);
                __m256d c_vec = _mm256_loadu_pd(&C[i * N + j]);

                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);

                _mm256_storeu_pd(&C[i * N + j], c_vec);
            }

            /*
                Resto escalar:
                Si N no es múltiplo de 4, calculamos los elementos restantes.
            */
            for (; j < N; j++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
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

    /*
        Memoria alineada a 32 bytes.
        AVX2 trabaja con registros de 256 bits = 32 bytes.
    */
    double *A;
    double *B;
    double *C;

    posix_memalign((void **)&A, 32, N * N * sizeof(double));
    posix_memalign((void **)&B, 32, N * N * sizeof(double));
    posix_memalign((void **)&C, 32, N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        printf("Error reservando memoria\n");
        return 1;
    }

    inicializar_matriz(A, N);
    inicializar_matriz(B, N);
    inicializar_cero(C, N);

    double inicio = omp_get_wtime();

    matmul_dlp(A, B, C, N);

    double fin = omp_get_wtime();

    printf("Tamaño matriz: %d x %d\n", N, N);
    printf("Tiempo DLP SIMD AVX2: %f segundos\n", fin - inicio);
    printf("Checksum: %f\n", checksum(C, N));

    free(A);
    free(B);
    free(C);

    return 0;
}