#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int values_per_process = 10;
    int numbers[values_per_process];

    srand((unsigned int)time(NULL) + rank * 37);
    for (int i = 0; i < values_per_process; i++) {
        numbers[i] = rand() % 1001;
    }

    int local_max = numbers[0];
    int local_min = numbers[0];
    for (int i = 1; i < values_per_process; i++) {
        if (numbers[i] > local_max) local_max = numbers[i];
        if (numbers[i] < local_min) local_min = numbers[i];
    }

    struct {
        int value;
        int rank;
    } local_max_pair, local_min_pair, global_max_pair, global_min_pair;

    local_max_pair.value = local_max;
    local_max_pair.rank = rank;
    local_min_pair.value = local_min;
    local_min_pair.rank = rank;

    MPI_Reduce(&local_max_pair, &global_max_pair, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min_pair, &global_min_pair, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global maximum value = %d from process %d\n", global_max_pair.value, global_max_pair.rank);
        printf("Global minimum value = %d from process %d\n", global_min_pair.value, global_min_pair.rank);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
