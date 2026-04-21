/*
 * Question 5: Find all perfect numbers up to a maximum value
 * Perfect number: equals the sum of its proper divisors
 * Example: 6 = 1 + 2 + 3 (proper divisors of 6)
 * Master-Slave Model using MPI_Recv with MPI_ANY_SOURCE
 * Master: distributes work, receives results
 * Slaves: requests work, tests for perfect numbers, returns results
 * Compile: mpicc -o q5 q5.c -lm
 * Run: mpirun -np 4 ./q5
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_VALUE 10000

// Function to check if a number is perfect
// A perfect number equals the sum of its proper divisors
int is_perfect(int n) {
    if (n <= 1) return 0;
    
    int sum = 1;  // Start with 1 (always a proper divisor for n > 1)
    
    // Find all divisors up to sqrt(n)
    int limit = (int)sqrt(n);
    for (int i = 2; i <= limit; i++) {
        if (n % i == 0) {
            sum += i;
            // Add complement divisor if it's different from i
            if (i != n / i) {
                sum += n / i;
            }
        }
    }
    
    return (sum == n);
}

// Calculate sum of proper divisors (for verification)
int sum_of_divisors(int n) {
    if (n <= 1) return 0;
    
    int sum = 1;
    int limit = (int)sqrt(n);
    for (int i = 2; i <= limit; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i) {
                sum += n / i;
            }
        }
    }
    return sum;
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
    
    int num_perfect = 0;
    int *perfect_numbers = (int*)malloc(MAX_VALUE * sizeof(int));
    
    if (rank == 0) {
        // MASTER PROCESS
        printf("\n=== Perfect Number Finder using MPI ===\n");
        printf("Finding perfect numbers up to %d using %d processes\n\n", MAX_VALUE, size);
        
        MPI_Status status;
        int number_to_check = 2;
        int slave_msg;
        
        // Send initial work to each slave
        for (int i = 1; i < size; i++) {
            MPI_Send(&number_to_check, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            number_to_check++;
        }
        
        // Continue distributing work while there are numbers to check
        while (number_to_check <= MAX_VALUE) {
            // Wait for any slave to request work
            MPI_Recv(&slave_msg, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            int slave_rank = status.MPI_SOURCE;
            
            // If positive: it's a perfect number, add to list
            if (slave_msg > 0) {
                perfect_numbers[num_perfect++] = slave_msg;
                printf("Found perfect number: %d (divisor sum: %d)\n", slave_msg, sum_of_divisors(slave_msg));
            }
            // If negative: it's not perfect, ignore
            
            // Send next number to test
            MPI_Send(&number_to_check, 1, MPI_INT, slave_rank, 0, MPI_COMM_WORLD);
            number_to_check++;
        }
        
        // Collect final results from all slaves
        for (int i = 1; i < size; i++) {
            MPI_Recv(&slave_msg, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            if (slave_msg > 0) {
                perfect_numbers[num_perfect++] = slave_msg;
                printf("Found perfect number: %d (divisor sum: %d)\n", slave_msg, sum_of_divisors(slave_msg));
            }
        }
        
        // Print results
        printf("\n=====================================\n");
        printf("Total perfect numbers found: %d\n", num_perfect);
        for (int i = 0; i < num_perfect; i++) {
            printf("Perfect Number %d: %d\n", i+1, perfect_numbers[i]);
            printf("  Divisors: 1");
            for (int j = 2; j < perfect_numbers[i]/2 + 1; j++) {
                if (perfect_numbers[i] % j == 0) {
                    printf(", %d", j);
                }
            }
            printf(" (sum = %d)\n", sum_of_divisors(perfect_numbers[i]));
        }
        printf("=====================================\n\n");
        
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
            
            // Check if we're done (master sends value > MAX_VALUE)
            if (number > MAX_VALUE) break;
            
            // Test if perfect number
            if (is_perfect(number)) {
                result = number;  // Send positive if perfect
            } else {
                result = -number;  // Send negative if not perfect
            }
            
            // Send result to master
            MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    free(perfect_numbers);
    MPI_Finalize();
    return 0;
}
