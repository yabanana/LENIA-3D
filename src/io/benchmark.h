#pragma once

#include "core/config.h"

#include <string>
#include <vector>

struct BenchmarkResult {
    int dimension;
    int grid_size;
    int num_threads;
    int num_iterations;
    double total_time_ms;
    double avg_step_time_ms;
    double steps_per_second;
};

// Run benchmark: execute num_iterations steps and measure total elapsed time.
// Returns a single BenchmarkResult with timing statistics.
BenchmarkResult run_benchmark(const LeniaConfig& config, int num_iterations = 100);

// Run scaling benchmark: test with different thread counts.
// For each thread count in the vector, runs a full benchmark and collects results.
std::vector<BenchmarkResult> run_scaling_benchmark(
    const LeniaConfig& base_config,
    const std::vector<int>& thread_counts,
    int num_iterations = 100);

// Print results to stdout in a formatted table.
void print_benchmark_results(const std::vector<BenchmarkResult>& results);

// Export results to CSV file.
void export_benchmark_csv(const std::vector<BenchmarkResult>& results,
                          const std::string& filename);
