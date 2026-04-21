/*
 * Question 3: Distributed Dot Product & Amdahl's Law
 * Task: Calculate dot product of two 500M-element vectors in parallel
 * Demonstrates MPI_Bcast and MPI_Reduce
 * Measures speedup and parallel efficiency
 * Compile: mpicc -o q3 q3.c -lm
 * Run: mpirun -np <num_processes> ./q3
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOTAL_SIZE 500000000LL  // 500 million elements

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    double multiplier = 2.5;  // Scaling multiplier
    
    // ========== BROADCAST: Rank 0 sends multiplier to all ==========
    if (rank == 0) {
        printf("\n=== Parallel Dot Product with Amdahl's Law Analysis ===\n");
        printf("Total Vector Size: %lld elements\n", TOTAL_SIZE);
        printf("Number of Processes: %d\n", size);
        printf("Multiplier: %.2f\n\n", multiplier);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = MPI_Wtime();
    
    // Broadcast multiplier
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // ========== LOCAL COMPUTATION: Each process computes its chunk ==========
    long long chunk = TOTAL_SIZE / size;
    long long start = rank * chunk;
    long long end = (rank == size - 1) ? TOTAL_SIZE : start + chunk;
    long long local_size = end - start;
    
    double *A = (double*)malloc(local_size * sizeof(double));
    double *B = (double*)malloc(local_size * sizeof(double));
    
    // Generate local vectors
    for (long long i = 0; i < local_size; i++) {
        A[i] = 1.0;
        B[i] = 2.0 * multiplier;
    }
    
    // Perform local dot product
    double local_dot = 0.0;
    for (long long i = 0; i < local_size; i++) {
        local_dot += A[i] * B[i];
    }
    
    // ========== REDUCE: Gather all results to rank 0 ==========
    double global_dot = 0.0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double total_end = MPI_Wtime();
    double parallel_time = total_end - total_start;
    
    if (rank == 0) {
        // Calculate sequential time (approximate)
        double seq_start = MPI_Wtime();
        double seq_dot = 0.0;
        for (long long i = 0; i < TOTAL_SIZE; i++) {
            seq_dot += 1.0 * (2.0 * multiplier);
        }
        double seq_end = MPI_Wtime();
        double seq_time = seq_end - seq_start;
        
        // Calculate metrics
        double speedup = seq_time / parallel_time;
        double efficiency = speedup / size;
        
        printf("Sequential Time:        %.6f seconds\n", seq_time);
        printf("Parallel Time (MPI):    %.6f seconds\n", parallel_time);
        printf("\nPerformance Metrics:\n");
        printf("Speedup (P=%d):         %.2f x\n", size, speedup);
        printf("Parallel Efficiency:    %.2f%% (%.3f/1)\n", efficiency * 100, efficiency);
        printf("Dot Product Result:     %.2e\n", global_dot);
        
        printf("\nAmdahl's Law Analysis:\n");
        printf("Perfect linear would give Speedup = %d\n", size);
        printf("Actual Speedup = %.2f (limited by communication & serialization)\n", speedup);
        printf("=========================================================\n\n");
    }
    
    free(A);
    free(B);
    MPI_Finalize();
    return 0;
}