#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SIZE 8

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int vector_size = VECTOR_SIZE;
    int A[VECTOR_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8};
    int B[VECTOR_SIZE] = {8, 7, 6, 5, 4, 3, 2, 1};
    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int base = vector_size / size;
        int remainder = vector_size % size;
        int offset = 0;
        for (int p = 0; p < size; p++) {
            sendcounts[p] = base + (p < remainder ? 1 : 0);
            displs[p] = offset;
            offset += sendcounts[p];
        }
    }

    int local_count;
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_A = malloc(local_count * sizeof(int));
    int *local_B = malloc(local_count * sizeof(int));
    if ((local_count > 0 && !local_A) || (local_count > 0 && !local_B)) {
        fprintf(stderr, "Memory allocation failed on process %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Scatterv(A, sendcounts, displs, MPI_INT,
                 local_A, local_count, MPI_INT,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, sendcounts, displs, MPI_INT,
                 local_B, local_count, MPI_INT,
                 0, MPI_COMM_WORLD);

    int local_dot = 0;
    for (int i = 0; i < local_count; i++) {
        local_dot += local_A[i] * local_B[i];
    }

    int global_dot;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Parallel dot product result = %d\n", global_dot);
        printf("Expected result = 120\n");
    }

    free(local_A);
    free(local_B);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
