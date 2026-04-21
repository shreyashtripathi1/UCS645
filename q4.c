/*
 * Question 4: Find all positive primes up to a maximum value
 * Master-Slave Model using MPI_Recv with MPI_ANY_SOURCE
 * Master: distributes work, receives results
 * Slaves: requests work, tests for primes, returns results
 * Compile: mpicc -o q4 q4.c -lm
 * Run: mpirun -np 4 ./q4
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_VALUE 100000

// Function to check if a number is prime
int is_prime(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    
    int limit = (int)sqrt(n) + 1;
    for (int i = 3; i <= limit; i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) {
        if (rank == 0) printf("Error: Need at least 2 processes (1 master + 1 slave)\n");
        MPI_Finalize();
        return 1;
    }
    
    int num_primes = 0;
    int *primes = (int*)malloc(MAX_VALUE * sizeof(int));
    
    if (rank == 0) {
        // MASTER PROCESS
        printf("\n=== Prime Number Finder using MPI ===\n");
        printf("Finding primes up to %d using %d processes\n\n", MAX_VALUE, size);
        
        MPI_Status status;
        int number_to_check = 2;
        int slave_msg;
        
        // Main loop: distribute work and collect results
        for (int i = 1; i < size; i++) {
            // Send initial work to each slave
            MPI_Send(&number_to_check, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            number_to_check++;
        }
        
        // Continue distributing work while there are numbers to check
        while (number_to_check <= MAX_VALUE) {
            // Wait for any slave to request work
            MPI_Recv(&slave_msg, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int slave_rank = status.MPI_SOURCE;
            
            // If positive: it's a prime, add to list
            if (slave_msg > 0) {
                primes[num_primes++] = slave_msg;
            }
            // If negative: it's not prime, ignore
            
            // Send next number to test
            MPI_Send(&number_to_check, 1, MPI_INT, slave_rank, 0, MPI_COMM_WORLD);
            number_to_check++;
        }
        
        // Collect final results from all slaves
        for (int i = 1; i < size; i++) {
            MPI_Recv(&slave_msg, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            if (slave_msg > 0) {
                primes[num_primes++] = slave_msg;
            }
        }
        
        // Print results
        printf("Found %d prime numbers:\n", num_primes);
        for (int i = 0; i < num_primes && i < 100; i++) {
            if (i > 0 && i % 10 == 0) printf("\n");
            printf("%6d ", primes[i]);
        }
        if (num_primes > 100) printf("\n... and %d more", num_primes - 100);
        printf("\n=====================================\n\n");
        
    } else {
        // SLAVE PROCESS
        int number, result;
        MPI_Status status;
        
        // Request initial work (send 0 to indicate starting)
        MPI_Send(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        
        // Continuous work loop
        while (1) {
            // Receive number to test
            MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            
            // Check if we're done (master sends 0 after MAX_VALUE)
            if (number > MAX_VALUE) break;
            
            // Test if prime
            if (is_prime(number)) {
                result = number;  // Send positive if prime
            } else {
                result = -number;  // Send negative if not prime
            }
            
            // Send result to master
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    free(primes);
    MPI_Finalize();
    return 0;
}
