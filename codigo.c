// Versión base: mm_seq.c
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