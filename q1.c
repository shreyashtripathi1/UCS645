/*
 * Question 1: DAXPY Operation
 * Task: Implement an MPI program to complete the DAXPY operation.
 * Operation: X[i] = a*X[i] + Y[i]
 * Measure the speedup compared to the uniprocessor implementation
 * Compile: mpicc -o q1 q1.c -lm
 * Run: mpirun -np <num_processes> ./q1
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VECTOR_SIZE (1 << 16)  // 2^16 = 65536 elements

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Allocate local arrays
    int local_size = VECTOR_SIZE / size;
    if (VECTOR_SIZE % size != 0 && rank < VECTOR_SIZE % size) {
        local_size++;
    }
    
    double *X = (double*)malloc(local_size * sizeof(double));
    double *Y = (double*)malloc(local_size * sizeof(double));
    double a = 2.5;
    
    // Initialize arrays
    int global_start = (rank * VECTOR_SIZE) / size;
    for (int i = 0; i < local_size; i++) {
        X[i] = (double)(global_start + i);
        Y[i] = (double)(global_start + i) * 0.5;
    }
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    // Perform DAXPY operation: X[i] = a*X[i] + Y[i]
    for (int i = 0; i < local_size; i++) {
        X[i] = a * X[i] + Y[i];
    }
    
    // Synchronize and get end time
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    
    // Get max time across all processes
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("=== DAXPY Operation Results ===\n");
        printf("Vector Size: %d\n", VECTOR_SIZE);
        printf("Number of Processes: %d\n", size);
        printf("Local Vector Size per Process: %d\n", local_size);
        printf("Scalar a: %.2f\n", a);
        printf("Time taken (MPI): %.6f seconds\n", max_time);
    }
    
    // Calculate reference time for single process (only on rank 0)
    if (rank == 0) {
        double* X_seq = (double*)malloc(VECTOR_SIZE * sizeof(double));
        double* Y_seq = (double*)malloc(VECTOR_SIZE * sizeof(double));
        
        for (int i = 0; i < VECTOR_SIZE; i++) {
            X_seq[i] = (double)i;
            Y_seq[i] = (double)i * 0.5;
        }
        
        double seq_start = MPI_Wtime();
        for (int i = 0; i < VECTOR_SIZE; i++) {
            X_seq[i] = a * X_seq[i] + Y_seq[i];
        }
        double seq_end = MPI_Wtime();
        double seq_time = seq_end - seq_start;
        
        printf("Time taken (Sequential): %.6f seconds\n", seq_time);
        printf("Speedup: %.2f x\n", seq_time / max_time);
        printf("==============================\n\n");
        
        free(X_seq);
        free(Y_seq);
    }
    
    // Verify result (print sample from rank 0)
    if (rank == 0) {
        printf("Sample results from Process 0:\n");
        printf("Index 0: X[0] = %.2f (expected: %.2f)\n", X[0], a * 0.0 + 0.0 * 0.5);
        printf("Index 100: X[100] = %.2f (expected: %.2f)\n", X[100], a * 100.0 + 100.0 * 0.5);
    }
    
    free(X);
    free(Y);
    MPI_Finalize();
    return 0;
}