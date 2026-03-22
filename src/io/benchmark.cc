#include "io/benchmark.h"
#include "core/lenia.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// run_benchmark
// Creates a Lenia instance with the given config, initializes with random
// state, and runs num_iterations simulation steps while measuring wall-clock
// time using std::chrono::high_resolution_clock.
// ---------------------------------------------------------------------------
BenchmarkResult run_benchmark(const LeniaConfig& config, int num_iterations) {
    // Create the Lenia instance (this also builds the kernel FFT)
    Lenia lenia(config);
    lenia.init_random(config.seed);

    // Warm-up: run 2 steps to ensure caches and FFTW plans are primed
    lenia.step();
    lenia.step();

    // Timed region
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        lenia.step();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    BenchmarkResult result;
    result.dimension       = config.dimension;
    result.grid_size       = config.grid_size;
    result.num_threads     = config.num_threads;
    result.num_iterations  = num_iterations;
    result.total_time_ms   = elapsed_ms;
    result.avg_step_time_ms = elapsed_ms / static_cast<double>(num_iterations);
    result.steps_per_second = (result.avg_step_time_ms > 0.0)
                                  ? 1000.0 / result.avg_step_time_ms
                                  : 0.0;

    return result;
}

// ---------------------------------------------------------------------------
// run_scaling_benchmark
// Runs benchmarks across multiple thread counts. For each thread count,
// creates a modified config and calls run_benchmark.
// ---------------------------------------------------------------------------
std::vector<BenchmarkResult> run_scaling_benchmark(
    const LeniaConfig& base_config,
    const std::vector<int>& thread_counts,
    int num_iterations) {

    std::vector<BenchmarkResult> results;
    results.reserve(thread_counts.size());

    for (int tc : thread_counts) {
        LeniaConfig cfg = base_config;
        cfg.num_threads = tc;

        std::cout << "  Running benchmark with " << tc << " thread(s)..."
                  << std::flush;

        BenchmarkResult r = run_benchmark(cfg, num_iterations);
        results.push_back(r);

        std::cout << " done (" << std::fixed << std::setprecision(1)
                  << r.avg_step_time_ms << " ms/step)\n";
    }

    return results;
}

// ---------------------------------------------------------------------------
// print_benchmark_results
// Prints a formatted table of benchmark results to stdout.
// ---------------------------------------------------------------------------
void print_benchmark_results(const std::vector<BenchmarkResult>& results) {
    if (results.empty()) {
        std::cout << "No benchmark results to display.\n";
        return;
    }

    std::cout << "\n"
              << "=== Benchmark Results ===\n"
              << std::left
              << std::setw(6)  << "Dim"
              << std::setw(8)  << "Size"
              << std::setw(10) << "Threads"
              << std::setw(8)  << "Iters"
              << std::setw(14) << "Total(ms)"
              << std::setw(14) << "Avg(ms)"
              << std::setw(12) << "Steps/s"
              << "\n"
              << std::string(72, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(6)  << r.dimension
                  << std::setw(8)  << r.grid_size
                  << std::setw(10) << r.num_threads
                  << std::setw(8)  << r.num_iterations
                  << std::setw(14) << std::fixed << std::setprecision(1)
                  << r.total_time_ms
                  << std::setw(14) << std::fixed << std::setprecision(3)
                  << r.avg_step_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(1)
                  << r.steps_per_second
                  << "\n";
    }

    // Compute speedup relative to the first result (assumed single-threaded)
    if (results.size() > 1) {
        double base_time = results[0].avg_step_time_ms;
        std::cout << "\n--- Speedup (relative to " << results[0].num_threads
                  << " thread(s)) ---\n";
        for (const auto& r : results) {
            double speedup = (r.avg_step_time_ms > 0.0)
                                 ? base_time / r.avg_step_time_ms
                                 : 0.0;
            std::cout << "  " << r.num_threads << " thread(s): "
                      << std::fixed << std::setprecision(2)
                      << speedup << "x\n";
        }
    }

    std::cout << "=========================\n\n";
}

// ---------------------------------------------------------------------------
// export_benchmark_csv
// Writes benchmark results to a CSV file with a header row.
// ---------------------------------------------------------------------------
void export_benchmark_csv(const std::vector<BenchmarkResult>& results,
                          const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    out << "dimension,grid_size,num_threads,num_iterations,"
        << "total_time_ms,avg_step_time_ms,steps_per_second\n";

    out << std::fixed << std::setprecision(6);

    for (const auto& r : results) {
        out << r.dimension << ','
            << r.grid_size << ','
            << r.num_threads << ','
            << r.num_iterations << ','
            << r.total_time_ms << ','
            << r.avg_step_time_ms << ','
            << r.steps_per_second << '\n';
    }
}
