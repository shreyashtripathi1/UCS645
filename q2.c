/*
 * Question 2: The Broadcast Race (Linear vs. Tree Communication)
 * Task: Compare manual for-loop broadcast (MyBcast) vs MPI_Bcast
 * MyBcast: Rank 0 uses a for loop with MPI_Send
 * MPI_Bcast: Uses optimized tree-based communication
 * Compile: mpicc -o q2 q2.c
 * Run: mpirun -np <num_processes> ./q2
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 10000000  // 10 million doubles = ~80 MB

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    double *array = (double*)malloc(ARRAY_SIZE * sizeof(double));
    
    // Initialize array on rank 0
    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            array[i] = (double)i * 1.5;
        }
    }
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ========== PART A: Manual Broadcast (MyBcast) ==========
    double mybcast_start = MPI_Wtime();
    
    if (rank == 0) {
        // Rank 0 sends to all other ranks using a for loop
        for (int i = 1; i < size; i++) {
            MPI_Send(array, ARRAY_SIZE, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // All other ranks receive from rank 0
        MPI_Recv(array, ARRAY_SIZE, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    double mybcast_end = MPI_Wtime();
    double mybcast_time = mybcast_end - mybcast_start;
    
    // Synchronize
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Reset array for MPI_Bcast test
    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            array[i] = (double)i * 1.5;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ========== PART B: Optimized MPI_Bcast ==========
    double bcast_start = MPI_Wtime();
    MPI_Bcast(array, ARRAY_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double bcast_end = MPI_Wtime();
    double bcast_time = bcast_end - bcast_start;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Print results
    if (rank == 0) {
        printf("\n=== Broadcast Race Results ===\n");
        printf("Array Size: %d (%.1f MB)\n", ARRAY_SIZE, ARRAY_SIZE * 8.0 / 1e6);
        printf("Number of Processes: %d\n", size);
        printf("\nMyBcast (Linear, for-loop): %.6f seconds\n", mybcast_time);
        printf("MPI_Bcast (Tree-based):    %.6f seconds\n", bcast_time);
        printf("Speedup Factor:             %.2f x\n", mybcast_time / bcast_time);
        printf("\nAnalysis:\n");
        printf("- MyBcast Time Complexity: O(N) where N = number of processes\n");
        printf("- MPI_Bcast Time Complexity: O(log N) with tree-based algorithm\n");
        printf("==============================\n\n");
    }
    
    free(array);
    MPI_Finalize();
    return 0;
}