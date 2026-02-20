#include "correlate.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstdlib>
#include <omp.h>


// Generate random number in [-1, 1]
double generate_random_number()
{
    static thread_local std::mt19937 mt(std::random_device{}());
    static thread_local std::uniform_real_distribution<double> range(-1.0, 1.0);

    return range(mt);
}


int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "Usage: ./correlate_matrix_sequential "
                  << "<number_of_rows> <number_of_columns>\n";
        return -1;
    }

    int rows = std::atoi(argv[1]);
    int cols = std::atoi(argv[2]);

    std::vector<std::vector<double> > matrix(
        rows, std::vector<double>(cols)
    );

    std::vector<double> flat_array(rows * cols);


    // Fill matrix in parallel
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            double num = generate_random_number();

            matrix[i][j] = num;
            flat_array[i * cols + j] = num;
        }
    }


    // Run sequential correlation
    std::chrono::duration<double, std::milli> seq_time =
        correlate_matrix_sequential(matrix);


    // Print result
    std::cout << "Execution time: "
              << std::fixed << std::setprecision(2)
              << seq_time.count()
              << " ms\n";


    return 0;
}
