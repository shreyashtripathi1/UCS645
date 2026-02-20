#pragma once

#include <vector>    
#include <chrono>
#include <cstddef>   


std::chrono::duration<double, std::milli>
correlate_matrix_sequential(
    const std::vector<std::vector<double> >& data
);


std::chrono::duration<double, std::milli>
correlate_matrix_parallel_2d_array(
    const std::vector<std::vector<double> >& data
);


std::chrono::duration<double, std::milli>
correlate_matrix_parallel_flat_array(
    const std::vector<double>& data,
    size_t rows,
    size_t cols
);
