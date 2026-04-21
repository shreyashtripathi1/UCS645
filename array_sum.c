#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int total_elements = 100;
    int *array = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        array = malloc(total_elements * sizeof(int));
        if (!array) {
            fprintf(stderr, "Memory allocation failed on root process.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for (int i = 0; i < total_elements; i++) {
            array[i] = i + 1;
        }

        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int base = total_elements / size;
        int remainder = total_elements % size;
        int offset = 0;
        for (int p = 0; p < size; p++) {
            sendcounts[p] = base + (p < remainder ? 1 : 0);
            displs[p] = offset;
            offset += sendcounts[p];
        }
    }

    int local_count;
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_array = malloc(local_count * sizeof(int));
    if (local_count > 0 && !local_array) {
        fprintf(stderr, "Memory allocation failed on process %d.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Scatterv(array, sendcounts, displs, MPI_INT,
                 local_array, local_count, MPI_INT,
                 0, MPI_COMM_WORLD);

    int local_sum = 0;
    for (int i = 0; i < local_count; i++) {
        local_sum += local_array[i];
    }

    int global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double average = (double)global_sum / total_elements;
        printf("Global sum = %d\n", global_sum);
        printf("Expected sum = 5050\n");
        printf("Average value = %.2f\n", average);
    }

    free(local_array);
    free(array);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
