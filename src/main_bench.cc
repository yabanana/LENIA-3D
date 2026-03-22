#include "core/config.h"
#include "core/lenia.h"
#include "io/benchmark.h"

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// parse_int_list
// Parses a comma-separated string of integers, e.g. "1,2,4,8"
// ---------------------------------------------------------------------------
static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> result;
    std::istringstream stream(s);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            result.push_back(std::stoi(token));
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    // Parse Lenia config from common args (--dim, --size, --mu, etc.)
    LeniaConfig config = parse_args(argc, argv);

    // Parse benchmark-specific args: --threads and --iterations
    // These override/extend the base config parsing.
    std::vector<int> thread_counts;
    int num_iterations = 100;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            // Check if the value contains a comma (list format)
            std::string val = argv[i + 1];
            if (val.find(',') != std::string::npos) {
                thread_counts = parse_int_list(val);
            } else {
                // Single thread count was already handled by parse_args;
                // but for benchmark mode we treat it as a single-element list
                thread_counts.push_back(std::stoi(val));
            }
            ++i;
        } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            num_iterations = std::stoi(argv[i + 1]);
            ++i;
        }
    }

    // Default thread counts if none specified
    if (thread_counts.empty()) {
        thread_counts = {1, 2, 4, 8};
    }

    std::cout << "=== Lenia Scaling Benchmark ===\n";
    print_config(config);

    std::cout << "Thread counts: ";
    for (std::size_t i = 0; i < thread_counts.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << thread_counts[i];
    }
    std::cout << "\nIterations per run: " << num_iterations << "\n\n";

    // Run the scaling benchmark
    auto results = run_scaling_benchmark(config, thread_counts, num_iterations);
    print_benchmark_results(results);

    // Export CSV
    std::string csv_file = "benchmark_" + std::to_string(config.dimension) + "d_"
                          + std::to_string(config.grid_size) + ".csv";
    export_benchmark_csv(results, csv_file);
    std::cout << "Results exported to: " << csv_file << "\n";

    return 0;
}
