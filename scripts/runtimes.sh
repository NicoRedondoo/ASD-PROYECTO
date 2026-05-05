#!/bin/bash

N=1000
REPS=3
OUT="csv/resultados.csv"

echo "Version,Optimizacion,Compilacion,N,Procesos,Hilos,Repeticion,Tiempo" > $OUT

echo "Compilando..."

gcc codigo.c -O0 -fopenmp -o codigo_O0
gcc codigo.c -O2 -fopenmp -o codigo_O2

gcc codigo_optimizado_ilp.c -O0 -fopenmp -o matmul_ilp_O0
gcc codigo_optimizado_ilp.c -O2 -fopenmp -o matmul_ilp_O2

gcc codigo_optimizado_dlp.c -O0 -mavx2 -mfma -fopenmp -o matmul_dlp_O0
gcc codigo_optimizado_dlp.c -O2 -mavx2 -mfma -fopenmp -o matmul_dlp_O2

gcc codigo_optimizado_tlp.c -O0 -fopenmp -o matmul_tlp_O0
gcc codigo_optimizado_tlp.c -O2 -fopenmp -o matmul_tlp_O2

mpicc codigo_optimizado_mpi.c -O0 -o matmul_mpi_O0
mpicc codigo_optimizado_mpi.c -O2 -o matmul_mpi_O2

mpicc codigo_optimizado_mpi_openmp.c -O0 -fopenmp -o matmul_mpi_omp_O0
mpicc codigo_optimizado_mpi_openmp.c -O2 -fopenmp -o matmul_mpi_omp_O2

echo "Ejecutando pruebas..."

extraer_tiempo() {
    grep -oE "[0-9]+\.[0-9]+" | head -n 1
}

run_normal() {
    VERSION=$1
    OPT=$2
    COMP=$3
    EXEC=$4

    for r in $(seq 1 $REPS); do
        T=$(./$EXEC $N | grep "Tiempo" | grep -oE "[0-9]+\.[0-9]+")
        echo "$VERSION,$OPT,$COMP,$N,1,1,$r,$T" >> $OUT
    done
}

run_openmp() {
    VERSION=$1
    OPT=$2
    COMP=$3
    EXEC=$4

    for h in 1 2 4 8; do
        export OMP_NUM_THREADS=$h

        for r in $(seq 1 $REPS); do
            T=$(./$EXEC $N | grep "Tiempo" | grep -oE "[0-9]+\.[0-9]+")
            echo "$VERSION,$OPT,$COMP,$N,1,$h,$r,$T" >> $OUT
        done
    done
}

run_mpi() {
    VERSION=$1
    OPT=$2
    COMP=$3
    EXEC=$4

    for p in 1 2 4; do
        for r in $(seq 1 $REPS); do
            T=$(mpirun -np $p ./$EXEC $N | grep "Tiempo" | grep -oE "[0-9]+\.[0-9]+")
            echo "$VERSION,$OPT,$COMP,$N,$p,1,$r,$T" >> $OUT
        done
    done
}

run_mpi_openmp() {
    VERSION=$1
    OPT=$2
    COMP=$3
    EXEC=$4

    for p in 1 2 4; do
        for h in 1 2 4; do
            export OMP_NUM_THREADS=$h

            for r in $(seq 1 $REPS); do
                T=$(mpirun -np $p ./$EXEC $N | grep "Tiempo" | grep -oE "[0-9]+\.[0-9]+")
                echo "$VERSION,$OPT,$COMP,$N,$p,$h,$r,$T" >> $OUT
            done
        done
    done
}

run_normal "Secuencial" "Base" "O0" "codigo_O0"
run_normal "Secuencial" "Base" "O2" "codigo_O2"

run_normal "ILP" "Desenrollado" "O0" "codigo_optimizado_ilp_O0"
run_normal "ILP" "Desenrollado" "O2" "codigo_optimizado_ilp_O2"

run_normal "DLP" "SIMD AVX2" "O0" "codigo_optimizado_dlp_O0"
run_normal "DLP" "SIMD AVX2" "O2" "codigo_optimizado_dlp_O2"

run_openmp "TLP" "OpenMP" "O0" "codigo_optimizado_tlp_O0"
run_openmp "TLP" "OpenMP" "O2" "codigo_optimizado_tlp_O2"

run_mpi "MPI" "Distribuido" "O0" "codigo_optimizado_mpi_O0"
run_mpi "MPI" "Distribuido" "O2" "codigo_optimizado_mpi_O2"

run_mpi_openmp "MPI+OpenMP" "Hibrido" "O0" "codigo_optimizado_mpi_openmp_O0"
run_mpi_openmp "MPI+OpenMP" "Hibrido" "O2" "codigo_optimizado_mpi_openmp_O2"

echo "Listo. Resultados guardados en $OUT"