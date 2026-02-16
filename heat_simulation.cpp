#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <string>

const int N = 2000;
const int STEPS = 500;
const double ALPHA = 0.01;
const double DX = 1.0;
const double DT = 0.1;
const int CHUNK_SIZE = 64;

struct SimResult {
    double duration_ms;
    double total_energy;
};

struct ScheduleConfig {
    omp_sched_t type;
    int chunk_size;
    std::string name;
};

SimResult run_simulation(int num_threads,
                         omp_sched_t sched_type,
                         int chunk_size) {

    std::vector<double> T(N * N, 0.0);
    std::vector<double> T_next(N * N, 0.0);

    int center = N / 2;
    int radius = N / 10;

    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {

            int dx = i - center;
            int dy = j - center;

            if (dx*dx + dy*dy < radius*radius) {
                T[i * N + j] = 100.0;
            } else {
                T[i * N + j] = 0.0;
            }
        }
    }

    double cx = ALPHA * DT / (DX * DX);
    double cy = ALPHA * DT / (DX * DX);

    omp_set_num_threads(num_threads);
    omp_set_schedule(sched_type, chunk_size);

    auto start =
        std::chrono::high_resolution_clock::now();

    for (int step = 0; step < STEPS; ++step) {

        #pragma omp parallel for schedule(runtime)
        for (int i = 1; i < N - 1; ++i) {

            for (int j = 1; j < N - 1; ++j) {

                int idx   = i * N + j;
                int up    = (i - 1) * N + j;
                int down  = (i + 1) * N + j;
                int left  = i * N + (j - 1);
                int right = i * N + (j + 1);

                T_next[idx] =
                    T[idx]
                    + cx * (T[up]   - 2*T[idx] + T[down])
                    + cy * (T[left] - 2*T[idx] + T[right]);
            }
        }

        std::swap(T, T_next);
    }

    auto end =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli>
        duration = end - start;

    double total_energy = 0.0;

    #pragma omp parallel for reduction(+:total_energy) num_threads(num_threads)
    for (int i = 0; i < N * N; ++i) {
        total_energy += T[i];
    }

    SimResult res;
    res.duration_ms = duration.count();
    res.total_energy = total_energy;

    return res;
}

int main() {

    std::cout << "With No Threading\n";

    SimResult seq_res =
        run_simulation(1, omp_sched_static, 0);

    std::cout << "Execution time: "
              << std::fixed << std::setprecision(2)
              << seq_res.duration_ms << " ms\n";

    std::cout << "Total Energy: "
              << std::scientific
              << seq_res.total_energy << "\n";


    std::vector<ScheduleConfig> strategies;

    strategies.push_back(
        { omp_sched_static, 0, "STATIC" });

    strategies.push_back(
        { omp_sched_dynamic, CHUNK_SIZE,
          "DYNAMIC (Chunk " + std::to_string(CHUNK_SIZE) + ")" });

    strategies.push_back(
        { omp_sched_guided, CHUNK_SIZE,
          "GUIDED (Chunk " + std::to_string(CHUNK_SIZE) + ")" });


    int max_threads = omp_get_max_threads();
    if (max_threads < 2) max_threads = 12;


    for (int t = 2; t <= max_threads; t += 2) {

        std::cout << "\nWith " << t << " threads\n";

        for (size_t i = 0; i < strategies.size(); ++i) {

            const ScheduleConfig& strat = strategies[i];

            SimResult par_res =
                run_simulation(t,
                               strat.type,
                               strat.chunk_size);

            double speed_up =
                seq_res.duration_ms /
                par_res.duration_ms;

            double ops = (double)N * N * STEPS;

            double throughput =
                ops / (par_res.duration_ms / 1000.0) / 1e6;

            double efficiency =
                speed_up / t;


            std::cout << "[ "
                      << strat.name
                      << " ]\n";

            std::cout << "Execution time: "
                      << std::fixed << std::setprecision(2)
                      << par_res.duration_ms << " ms\n";

            std::cout << "Speed Up: "
                      << speed_up << "x\n";

            std::cout << "Throughput: "
                      << throughput << "\n";

            std::cout << "Efficiency: "
                      << efficiency << "\n";
        }
    }

    return 0;
}
