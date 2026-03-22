#include "config.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// Helper: read next argv value or exit on missing argument
// ---------------------------------------------------------------------------
static const char* require_arg(int i, int argc, char* argv[]) {
    if (i + 1 >= argc) {
        std::cerr << "Error: option " << argv[i] << " requires a value\n";
        std::exit(1);
    }
    return argv[i + 1];
}

// ---------------------------------------------------------------------------
// Helper: trim leading and trailing whitespace from a string
// ---------------------------------------------------------------------------
static std::string trim(const std::string& s) {
    std::size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    std::size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// ---------------------------------------------------------------------------
// Helper: parse comma-separated beta weights string into vector
// ---------------------------------------------------------------------------
static std::vector<float> parse_beta(const std::string& s) {
    std::vector<float> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        token = trim(token);
        if (!token.empty()) {
            result.push_back(std::stof(token));
        }
    }
    if (result.empty()) {
        result.push_back(1.0f);
    }
    return result;
}

// ---------------------------------------------------------------------------
// apply_312_preset — Chan's documented 3D Lenia patterns (2 kernels)
// ---------------------------------------------------------------------------
void apply_312_preset(LeniaConfig& cfg, const std::string& name) {
    cfg.dimension = 3;

    if (name == "fish") {
        // Chan 312.json fish (kn=1): K0 R=20 b=[1] mu=0.13 s=0.025
        cfg.kernel = {20, {1.0f}, 4, KernelCoreFunc::Exponential};
        cfg.growth = {0.13f, 0.025f};
        cfg.T = 10;
        // K1: R=20 b=[1,3/4,1/12] mu=0.24 s=0.046
        KernelGrowthPair k1;
        k1.kernel = {20, {1.0f, 0.75f, 1.0f / 12.0f}, 4, KernelCoreFunc::Exponential};
        k1.growth = {0.24f, 0.046f};
        cfg.extra_kernels.clear();
        cfg.extra_kernels.push_back(k1);
    } else if (name == "divide") {
        // Chan 312.json divide (kn=1): K0 R=16 b=[1] mu=0.13 s=0.025
        cfg.kernel = {16, {1.0f}, 4, KernelCoreFunc::Exponential};
        cfg.growth = {0.13f, 0.025f};
        cfg.T = 10;
        // K1: R=16 b=[1,3/4,1/12] mu=0.24 s=0.046
        KernelGrowthPair k1;
        k1.kernel = {16, {1.0f, 0.75f, 1.0f / 12.0f}, 4, KernelCoreFunc::Exponential};
        k1.growth = {0.24f, 0.046f};
        cfg.extra_kernels.clear();
        cfg.extra_kernels.push_back(k1);
    } else if (name == "embryo") {
        // Chan 312.json embryo (kn=1): K0 R=15 b=[1] mu=0.1 s=0.0239
        cfg.kernel = {15, {1.0f}, 4, KernelCoreFunc::Exponential};
        cfg.growth = {0.1f, 0.0239f};
        cfg.T = 10;
        // K1: R=15 b=[1,1,0] mu=0.235 s=0.051
        KernelGrowthPair k1;
        k1.kernel = {15, {1.0f, 1.0f, 0.0f}, 4, KernelCoreFunc::Exponential};
        k1.growth = {0.235f, 0.051f};
        cfg.extra_kernels.clear();
        cfg.extra_kernels.push_back(k1);
    } else if (name == "ghost") {
        // 312.json #1 (kn=1): K0 R=20 b=[1] mu=0.073 s=0.0186
        cfg.kernel = {20, {1.0f}, 4, KernelCoreFunc::Exponential};
        cfg.growth = {0.073f, 0.0186f};
        cfg.T = 10;
        // K1: R=20 b=[1,5/6,0] mu=0.218 s=0.0505
        KernelGrowthPair k1;
        k1.kernel = {20, {1.0f, 5.0f / 6.0f, 0.0f}, 4, KernelCoreFunc::Exponential};
        k1.growth = {0.218f, 0.0505f};
        cfg.extra_kernels.clear();
        cfg.extra_kernels.push_back(k1);
    } else if (name == "exotic") {
        // 312.json #20 (kn=1): K0 R=15 b=[1] mu=0.35 s=0.068
        cfg.kernel = {15, {1.0f}, 4, KernelCoreFunc::Exponential};
        cfg.growth = {0.35f, 0.068f};
        cfg.T = 10;
        // K1: R=15 b=[7/12,7/12,1] mu=0.16 s=0.032
        KernelGrowthPair k1;
        k1.kernel = {15, {7.0f / 12.0f, 7.0f / 12.0f, 1.0f}, 4, KernelCoreFunc::Exponential};
        k1.growth = {0.16f, 0.032f};
        cfg.extra_kernels.clear();
        cfg.extra_kernels.push_back(k1);
    } else if (name == "protoeel") {
        // 312.json #36 proto-eel (kn=1): K0 R=15 b=[1,2/3] mu=0.13 s=0.018
        cfg.kernel = {15, {1.0f, 2.0f / 3.0f}, 4, KernelCoreFunc::Exponential};
        cfg.growth = {0.13f, 0.018f};
        cfg.T = 10;
        // K1: R=15 b=[1,1/4] mu=0.19 s=0.017
        KernelGrowthPair k1;
        k1.kernel = {15, {1.0f, 0.25f}, 4, KernelCoreFunc::Exponential};
        k1.growth = {0.19f, 0.017f};
        cfg.extra_kernels.clear();
        cfg.extra_kernels.push_back(k1);
    } else if (name == "butterfly") {
        // Chan 312.json butterfly (kn=1): K0 R=13 b=[11/12,1] mu=0.21 s=0.032
        cfg.kernel = {13, {11.0f / 12.0f, 1.0f}, 4, KernelCoreFunc::Exponential};
        cfg.growth = {0.21f, 0.032f};
        cfg.T = 10;
        // K1: R=13 b=[1/3,1/3,1] mu=0.12 s=0.021
        KernelGrowthPair k1;
        k1.kernel = {13, {1.0f / 3.0f, 1.0f / 3.0f, 1.0f}, 4, KernelCoreFunc::Exponential};
        k1.growth = {0.12f, 0.021f};
        cfg.extra_kernels.clear();
        cfg.extra_kernels.push_back(k1);
    } else {
        std::cerr << "Warning: unknown 312 preset '" << name << "'\n";
    }
}

// ---------------------------------------------------------------------------
// parse_args
// ---------------------------------------------------------------------------
LeniaConfig parse_args(int argc, char* argv[]) {
    LeniaConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (std::strcmp(arg, "--dim") == 0) {
            cfg.dimension = std::atoi(require_arg(i, argc, argv));
            ++i;
        } else if (std::strcmp(arg, "--size") == 0) {
            cfg.grid_size = std::atoi(require_arg(i, argc, argv));
            ++i;
        } else if (std::strcmp(arg, "--T") == 0) {
            cfg.T = std::atoi(require_arg(i, argc, argv));
            ++i;
        } else if (std::strcmp(arg, "--threads") == 0) {
            cfg.num_threads = std::atoi(require_arg(i, argc, argv));
            ++i;
        } else if (std::strcmp(arg, "--radius") == 0) {
            cfg.kernel.radius = std::atoi(require_arg(i, argc, argv));
            ++i;
        } else if (std::strcmp(arg, "--mu") == 0) {
            cfg.growth.mu = static_cast<float>(
                std::atof(require_arg(i, argc, argv)));
            ++i;
        } else if (std::strcmp(arg, "--sigma") == 0) {
            cfg.growth.sigma = static_cast<float>(
                std::atof(require_arg(i, argc, argv)));
            ++i;
        } else if (std::strcmp(arg, "--seed") == 0) {
            cfg.seed = static_cast<unsigned>(
                std::atoi(require_arg(i, argc, argv)));
            ++i;
        } else if (std::strcmp(arg, "--beta") == 0) {
            cfg.kernel.beta = parse_beta(require_arg(i, argc, argv));
            ++i;
        } else if (std::strcmp(arg, "--preset") == 0) {
            apply_312_preset(cfg, require_arg(i, argc, argv));
            ++i;
        } else if (std::strcmp(arg, "--pattern") == 0) {
            // Pattern is not stored in LeniaConfig but consumed by main().
            ++i;
        } else if (std::strcmp(arg, "--cells") == 0) {
            // Cells file path consumed by main().
            ++i;
        } else if (std::strcmp(arg, "--steps") == 0) {
            // Steps count is not stored in LeniaConfig; consumed by main().
            ++i;
        } else if (std::strcmp(arg, "--benchmark") == 0) {
            // Flag consumed by main(), no value.
        } else if (std::strcmp(arg, "--headless") == 0) {
            // Flag consumed by main(), no value.
        } else if (std::strcmp(arg, "--mass-log") == 0) {
            // Path consumed by main().
            ++i;
        } else if (std::strcmp(arg, "--snapshot") == 0) {
            // Path consumed by main().
            ++i;
        } else if (std::strcmp(arg, "--config") == 0) {
            // Load from config file, then continue with remaining args
            const char* path = require_arg(i, argc, argv);
            cfg = load_config_file(path);
            ++i;
        } else if (std::strcmp(arg, "--help") == 0
                   || std::strcmp(arg, "-h") == 0) {
            std::cout
                << "Usage: lenia [options]\n"
                << "\n"
                << "Simulation parameters:\n"
                << "  --dim N          Dimension: 2 or 3 (default: 2)\n"
                << "  --size N         Grid side length (default: 256)\n"
                << "  --T N            Time resolution, dt=1/T (default: 10)\n"
                << "  --threads N      OpenMP thread count (default: 1)\n"
                << "  --seed N         Random seed (default: 42)\n"
                << "\n"
                << "Kernel parameters:\n"
                << "  --radius N       Kernel radius R (default: 13)\n"
                << "  --beta S         Comma-separated ring weights (default: \"1\")\n"
                << "\n"
                << "Growth function parameters:\n"
                << "  --mu F           Growth center (default: 0.15)\n"
                << "  --sigma F        Growth width (default: 0.015)\n"
                << "\n"
                << "Initialization:\n"
                << "  --pattern S      Initial pattern: orbium, random, blob (default: orbium)\n"
                << "  --preset S       Chan 312 preset: fish, divide, embryo\n"
                << "\n"
                << "Execution:\n"
                << "  --steps N        Number of simulation steps (default: 1000)\n"
                << "  --benchmark      Run in benchmark mode (timing only)\n"
                << "  --headless       Run without graphical display\n"
                << "  --mass-log FILE  Write mass CSV (step,mass) in headless mode\n"
                << "  --snapshot FILE  Save final grid as binary in headless mode\n"
                << "  --config FILE    Load settings from a config file\n"
                << "  --help           Show this help message and exit\n";
            std::exit(0);
        } else {
            std::cerr << "Warning: unknown option '" << arg << "' (ignored)\n";
        }
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// load_config_file
// Reads a simple key=value text file. Lines starting with '#' are comments.
// Blank lines are skipped. Unknown keys are silently ignored.
// ---------------------------------------------------------------------------
LeniaConfig load_config_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open config file '" << path << "'\n";
        std::exit(1);
    }

    LeniaConfig cfg;
    std::string line;
    int line_num = 0;

    while (std::getline(file, line)) {
        ++line_num;
        line = trim(line);

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Find '=' separator
        std::size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            std::cerr << "Warning: config line " << line_num
                      << ": missing '=' separator, skipping\n";
            continue;
        }

        std::string key   = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos + 1));

        if (key == "dim" || key == "dimension") {
            cfg.dimension = std::stoi(value);
        } else if (key == "size" || key == "grid_size") {
            cfg.grid_size = std::stoi(value);
        } else if (key == "T") {
            cfg.T = std::stoi(value);
        } else if (key == "threads" || key == "num_threads") {
            cfg.num_threads = std::stoi(value);
        } else if (key == "seed") {
            cfg.seed = static_cast<unsigned>(std::stoi(value));
        } else if (key == "radius") {
            cfg.kernel.radius = std::stoi(value);
        } else if (key == "mu") {
            cfg.growth.mu = std::stof(value);
        } else if (key == "sigma") {
            cfg.growth.sigma = std::stof(value);
        } else {
            // Unknown key -- silently ignore
        }
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// print_config
// ---------------------------------------------------------------------------
// Helper: format beta vector as string like "[1,0.75,0.083]"
static std::string beta_to_string(const std::vector<float>& beta) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < beta.size(); ++i) {
        if (i > 0) oss << ",";
        oss << beta[i];
    }
    oss << "]";
    return oss.str();
}

void print_config(const LeniaConfig& config) {
    std::cout << "=== Lenia Configuration ===\n"
              << "  Dimension:      " << config.dimension << "D\n"
              << "  Grid size:      " << config.grid_size << "\n"
              << "  T (dt=1/T):     " << config.T << "\n"
              << "  Threads:        " << config.num_threads << "\n"
              << "  Seed:           " << config.seed << "\n"
              << "  Kernels:        " << config.num_kernels() << "\n"
              << "  [K0] R=" << config.kernel.radius
              << " kn=" << static_cast<int>(config.kernel.core_func)
              << " b=" << beta_to_string(config.kernel.beta)
              << " mu=" << config.growth.mu
              << " sigma=" << config.growth.sigma << "\n";

    for (std::size_t i = 0; i < config.extra_kernels.size(); ++i) {
        const auto& kgp = config.extra_kernels[i];
        std::cout << "  [K" << (i + 1) << "] R=" << kgp.kernel.radius
                  << " kn=" << static_cast<int>(kgp.kernel.core_func)
                  << " b=" << beta_to_string(kgp.kernel.beta)
                  << " mu=" << kgp.growth.mu
                  << " sigma=" << kgp.growth.sigma << "\n";
    }

    std::cout << "===========================\n";
}
