#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

#define SQUARE(a) ((a)*(a))

const int TOTAL_PARTICLES = 10000;

const double EPSILON = 1.0;
const double SIGMA = 1.0;
const double SIGMASQ = SQUARE(SIGMA);

struct vec3_t {
    double x, y, z;

    vec3_t() : x(0), y(0), z(0) {}
    vec3_t(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    vec3_t operator+(const vec3_t& other) const {
        return vec3_t(x + other.x, y + other.y, z + other.z);
    }

    vec3_t operator-(const vec3_t& other) const {
        return vec3_t(x - other.x, y - other.y, z - other.z);
    }

    vec3_t& operator+=(const vec3_t& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    vec3_t operator*(double s) const {
        return vec3_t(x * s, y * s, z * s);
    }
};

std::ostream& operator<<(std::ostream& os, const vec3_t& v) {
    os << "(" << v.x << ")i + (" << v.y << ")j + (" << v.z << ")k";
    return os;
}

struct Particles {
    std::vector<vec3_t> position;
    std::vector<vec3_t> force;
};

double get_random_number() {
    static thread_local std::mt19937 mt(std::random_device{}());
    static thread_local std::uniform_real_distribution<double> dist(-5.0, 5.0);
    return dist(mt);
}

vec3_t generate_random_vector() {
    return vec3_t(
        get_random_number(),
        get_random_number(),
        get_random_number()
    );
}

void init_particles(Particles& particles) {

    int n = particles.position.size();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        particles.position[i] = generate_random_vector();
        particles.force[i] = vec3_t(0,0,0);
    }
}

std::chrono::duration<double, std::milli>
compute_potential_and_force_sequential(Particles& particles,
                                       double& total_energy) {

    total_energy = 0.0;
    int n = particles.position.size();

    std::cout << "Using No Threading\n";

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < n; i++) {

        vec3_t current_force(0,0,0);

        for (int j = 0; j < n; j++) {

            if (i == j) continue;

            vec3_t delta = particles.position[i] - particles.position[j];

            double r2 = SQUARE(delta.x) + SQUARE(delta.y) + SQUARE(delta.z);

            if (r2 < 1e-10) continue;

            double r2_inv = 1.0 / r2;
            double s2_inv = SIGMASQ * r2_inv;
            double s6_inv = SQUARE(s2_inv) * s2_inv;
            double s12_inv = SQUARE(s6_inv);

            double pair_energy =
                4.0 * EPSILON * (s12_inv - s6_inv);

            total_energy += pair_energy;

            double force_scalar =
                (24.0 * EPSILON * r2_inv) *
                (2.0 * s12_inv - s6_inv);

            vec3_t force_vec = delta * force_scalar;

            current_force += force_vec;
        }

        particles.force[i] = current_force;
    }

    auto end = std::chrono::steady_clock::now();

    auto ms = end - start;

    std::cout << "Execution time: " << ms.count() << " ms\n";
    std::cout << "Total Energy: " << total_energy * 0.5 << "\n";

    return ms;
}

std::chrono::duration<double, std::milli>
compute_potential_and_force_parallel(Particles& particles,
                                     double& total_energy,
                                     int num_threads) {

    total_energy = 0.0;
    int n = particles.position.size();

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        particles.force[i] = vec3_t(0,0,0);
    }

    std::cout << "With " << num_threads << " threads\n";

    auto start = std::chrono::steady_clock::now();

    #pragma omp parallel for reduction(+:total_energy) \
            schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < n; i++) {

        vec3_t current_force(0,0,0);

        for (int j = 0; j < n; j++) {

            if (i == j) continue;

            vec3_t delta = particles.position[i] - particles.position[j];

            double r2 = SQUARE(delta.x) + SQUARE(delta.y) + SQUARE(delta.z);

            if (r2 < 1e-10) continue;

            double r2_inv = 1.0 / r2;
            double s2_inv = SIGMASQ * r2_inv;
            double s6_inv = SQUARE(s2_inv) * s2_inv;
            double s12_inv = SQUARE(s6_inv);

            double pair_energy =
                4.0 * EPSILON * (s12_inv - s6_inv);

            total_energy += pair_energy;

            double force_scalar =
                (24.0 * EPSILON * r2_inv) *
                (2.0 * s12_inv - s6_inv);

            vec3_t force_vec = delta * force_scalar;

            current_force += force_vec;
        }

        particles.force[i] = current_force;
    }

    auto end = std::chrono::steady_clock::now();

    auto ms = end - start;

    std::cout << "Execution time: " << ms.count() << " ms\n";
    std::cout << "Total Energy: " << total_energy * 0.5 << "\n";

    return ms;
}

int main() {

    Particles particles;

    particles.position.resize(TOTAL_PARTICLES);
    particles.force.resize(TOTAL_PARTICLES);

    init_particles(particles);

    int max_threads = omp_get_max_threads();

    double total_energy = 0.0;

    auto seq_ms =
        compute_potential_and_force_sequential(particles, total_energy);

    std::cout << "\n";

    for (int t = 2; t <= max_threads; t += 2) {

        auto par_ms =
            compute_potential_and_force_parallel(particles,
                                                 total_energy,
                                                 t);

        double speed_up = seq_ms.count() / par_ms.count();

        std::cout << "Speed Up: " << speed_up << "x\n";
        std::cout << "Throughput: "
                  << TOTAL_PARTICLES / par_ms.count() << "\n";
        std::cout << "Efficiency: "
                  << speed_up / t << "\n\n";
    }

    return 0;
}
