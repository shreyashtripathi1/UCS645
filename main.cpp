#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstdlib>
#include <omp.h>

#include "correlate.hpp"

// C++11 style function declaration
double generate_random_number() {
    static thread_local std::mt19937 mt(std::random_device{}());
    static thread_local std::uniform_real_distribution<double> range(-1.0, 1.0);
    return range(mt);
}

int main(int argc, char** argv) {

    if (argc != 3) {
        std::cout << "Usage: ./correlate_matrix <number_of_rows> <number_of_columns>\n";
        return -1;
    }

    int rows = std::atoi(argv[1]);
    int cols = std::atoi(argv[2]);

    int max_threads = omp_get_max_threads();

    std::vector<std::vector<double> > matrix(
        rows, std::vector<double>(cols)
    );

    std::vector<double> flat_array(rows * cols);

    // Initialize matrix in parallel
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            double num = generate_random_number();

            matrix[i][j] = num;
            flat_array[i * cols + j] = num;
        }
    }

    /* ---------------- Sequential ---------------- */

    std::cout << "Using No Threading\n";

    auto seq_time = correlate_matrix_sequential(matrix);

    std::cout << "Execution time: "
              << std::fixed << std::setprecision(2)
              << seq_time.count()
              << " ms\n\n";


    /* ---------------- Parallel (2D Array) ---------------- */

    std::cout << "Using Threading with 2D Heap Allocated Array\n\n";

    for (int num_threads = 2;
         num_threads <= max_threads;
         num_threads += 2) {

        std::cout << "With " << num_threads << " threads\n";

        omp_set_num_threads(num_threads);

        auto unoptimized_par_time =
            correlate_matrix_parallel_2d_array(matrix);

        std::cout << "Execution time: "
                  << std::fixed << std::setprecision(2)
                  << unoptimized_par_time.count()
                  << " ms\n";

        std::cout << "Speed Up: "
                  << seq_time.count() / unoptimized_par_time.count()
                  << "x\n\n";
    }


    /* ---------------- Parallel (Flat Array) ---------------- */

    std::cout << "Using Threading with Flat Array\n\n";

    for (int num_threads = 2;
         num_threads <= max_threads;
         num_threads += 2) {

        std::cout << "With " << num_threads << " threads\n";

        omp_set_num_threads(num_threads);

        auto optimized_par_time =
            correlate_matrix_parallel_2d_array(matrix);

        std::cout << "Execution time: "
                  << std::fixed << std::setprecision(2)
                  << optimized_par_time.count()
                  << " ms\n";

        std::cout << "Speed Up: "
                  << seq_time.count() / optimized_par_time.count()
                  << "x\n\n";
    }

    return 0;
}
