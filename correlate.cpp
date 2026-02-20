#include "correlate.hpp"

#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <stdexcept>



std::chrono::duration<double, std::milli>
correlate_matrix_sequential(const std::vector<std::vector<double> >& data)
{
    std::chrono::high_resolution_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();

    if (data.empty())
        return std::chrono::duration<double, std::milli>(0);

    size_t rows = data.size();
    size_t cols = data[0].size();

    std::vector<double> means(rows, 0.0);
    std::vector<double> inv_norms(rows, 0.0);

    for (size_t i = 0; i < rows; ++i) {

        double sum = 0.0;

        for (size_t k = 0; k < cols; ++k)
            sum += data[i][k];

        means[i] = sum / cols;

        double sq_sum = 0.0;

        for (size_t k = 0; k < cols; ++k) {
            double diff = data[i][k] - means[i];
            sq_sum += diff * diff;
        }

        inv_norms[i] = (sq_sum > 0)
                          ? 1.0 / std::sqrt(sq_sum)
                          : 0.0;
    }

    std::vector<std::vector<double> >
        result(rows, std::vector<double>(rows));

    for (size_t i = 0; i < rows; ++i) {

        result[i][i] = 1.0;

        for (size_t j = i + 1; j < rows; ++j) {

            double dot_product = 0.0;

            for (size_t k = 0; k < cols; ++k) {
                dot_product +=
                    (data[i][k] - means[i]) *
                    (data[j][k] - means[j]);
            }

            double corr =
                dot_product * inv_norms[i] * inv_norms[j];

            result[i][j] = corr;
            result[j][i] = corr;
        }
    }

    std::chrono::high_resolution_clock::time_point end_time =
        std::chrono::high_resolution_clock::now();

    return end_time - start_time;
}




std::chrono::duration<double, std::milli>
correlate_matrix_parallel_2d_array(
    const std::vector<std::vector<double> >& data)
{
    std::chrono::high_resolution_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();

    if (data.empty())
        return std::chrono::duration<double, std::milli>(0);

    size_t rows = data.size();
    size_t cols = data[0].size();

    std::vector<double> means(rows);
    std::vector<double> inv_norms(rows);


    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {

        double sum = 0.0;

        for (size_t k = 0; k < cols; ++k)
            sum += data[i][k];

        means[i] = sum / cols;

        double sq_sum = 0.0;

        for (size_t k = 0; k < cols; ++k) {
            double diff = data[i][k] - means[i];
            sq_sum += diff * diff;
        }

        inv_norms[i] = (sq_sum > 0)
                          ? 1.0 / std::sqrt(sq_sum)
                          : 0.0;
    }


    std::vector<std::vector<double> >
        result(rows, std::vector<double>(rows));


    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < rows; ++i) {

        result[i][i] = 1.0;

        for (size_t j = i + 1; j < rows; ++j) {

            double dot_product = 0.0;

            for (size_t k = 0; k < cols; ++k) {
                dot_product +=
                    (data[i][k] - means[i]) *
                    (data[j][k] - means[j]);
            }

            double corr =
                dot_product * inv_norms[i] * inv_norms[j];

            result[i][j] = corr;
            result[j][i] = corr;
        }
    }


    std::chrono::high_resolution_clock::time_point end_time =
        std::chrono::high_resolution_clock::now();

    return end_time - start_time;
}




std::chrono::duration<double, std::milli>
correlate_matrix_parallel_flat_array(
    const std::vector<double>& data,
    size_t rows,
    size_t cols)
{
    std::chrono::high_resolution_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();


    if (data.size() != rows * cols) {
        throw std::invalid_argument(
            "Data size does not match rows * cols");
    }


    std::vector<double> means(rows);
    std::vector<double> inv_norms(rows);


    #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {

        double sum = 0.0;
        size_t row_offset = i * cols;

        for (size_t k = 0; k < cols; ++k)
            sum += data[row_offset + k];

        means[i] = sum / cols;

        double sq_sum = 0.0;

        for (size_t k = 0; k < cols; ++k) {
            double diff =
                data[row_offset + k] - means[i];
            sq_sum += diff * diff;
        }

        inv_norms[i] = (sq_sum > 0)
                          ? 1.0 / std::sqrt(sq_sum)
                          : 0.0;
    }


    std::vector<double> result(rows * rows);


    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < rows; ++i) {

        result[i * rows + i] = 1.0;

        size_t i_offset = i * cols;

        for (size_t j = i + 1; j < rows; ++j) {

            size_t j_offset = j * cols;

            double dot_product = 0.0;

            for (size_t k = 0; k < cols; ++k) {

                double val_i =
                    data[i_offset + k] - means[i];

                double val_j =
                    data[j_offset + k] - means[j];

                dot_product += val_i * val_j;
            }

            double corr =
                dot_product * inv_norms[i] * inv_norms[j];

            result[i * rows + j] = corr;
            result[j * rows + i] = corr;
        }
    }


    std::chrono::high_resolution_clock::time_point end_time =
        std::chrono::high_resolution_clock::now();

    return end_time - start_time;
}
