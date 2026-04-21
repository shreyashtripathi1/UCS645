#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "This program requires at least 2 MPI processes.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int next_rank = (rank + 1) % size;
    int prev_rank = (rank - 1 + size) % size;
    int value;

    if (rank == 0) {
        value = 100;
        printf("Process %d starting ring with initial value %d\n", rank, value);
        MPI_Send(&value, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
        MPI_Recv(&value, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received final value %d after completing ring\n", rank, value);
    } else {
        MPI_Recv(&value, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received value %d from process %d\n", rank, value, prev_rank);
        value += rank;
        printf("Process %d added its rank and sending value %d to process %d\n", rank, value, next_rank);
        MPI_Send(&value, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
