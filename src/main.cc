#include "core/config.h"
#include "core/lenia.h"
#include "viz/renderer.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Extract a string flag from argv. Returns default_val if not found.
// ---------------------------------------------------------------------------
static std::string extract_flag(int argc, char* argv[], const char* flag,
                                const char* default_val) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0 && i + 1 < argc) {
            return argv[i + 1];
        }
    }
    return default_val;
}

// ---------------------------------------------------------------------------
// Check if a boolean flag is present in argv.
// ---------------------------------------------------------------------------
static bool has_flag(int argc, char* argv[], const char* flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Load cells from binary file into 3D grid.
// Format: 3 x int32 (nz, nr, nc) then float32 data [nz * nr * nc]
// ---------------------------------------------------------------------------
static bool load_cells_into_grid(const std::string& path, Grid<3>& grid, int N) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open cells file '" << path << "'\n";
        return false;
    }
    int32_t nz, nr, nc;
    fin.read(reinterpret_cast<char*>(&nz), 4);
    fin.read(reinterpret_cast<char*>(&nr), 4);
    fin.read(reinterpret_cast<char*>(&nc), 4);
    std::vector<float> cells(nz * nr * nc);
    fin.read(reinterpret_cast<char*>(cells.data()), cells.size() * sizeof(float));
    fin.close();

    grid.fill(0.0f);
    int od = (N - nz) / 2, orow = (N - nr) / 2, oc = (N - nc) / 2;
    for (int d = 0; d < nz; ++d)
        for (int r = 0; r < nr; ++r)
            for (int c = 0; c < nc; ++c) {
                float v = cells[d * nr * nc + r * nc + c];
                if (v > 0.0f)
                    grid.at(od + d, orow + r, oc + c) = v;
            }
    std::cout << "Loaded cells from " << path
              << " (" << nz << "x" << nr << "x" << nc << ")\n";
    return true;
}

// ---------------------------------------------------------------------------
// Save 3D grid state to binary file.
// Format: 3 x int32 (N, N, N) then float32 data [N^3]
// ---------------------------------------------------------------------------
static bool save_snapshot(const Grid<3>& grid, int N, const std::string& path) {
    std::ofstream fout(path, std::ios::binary);
    if (!fout.is_open()) {
        std::cerr << "Error: cannot open snapshot file '" << path << "'\n";
        return false;
    }
    int32_t dim = static_cast<int32_t>(N);
    fout.write(reinterpret_cast<const char*>(&dim), 4);
    fout.write(reinterpret_cast<const char*>(&dim), 4);
    fout.write(reinterpret_cast<const char*>(&dim), 4);
    fout.write(reinterpret_cast<const char*>(grid.data()),
               static_cast<std::streamsize>(N) * N * N * sizeof(float));
    fout.close();
    std::cout << "Saved snapshot to " << path << " (" << N << "^3)\n";
    return true;
}

// ---------------------------------------------------------------------------
// Run headless simulation: step N times, optionally log mass and save snapshot.
// ---------------------------------------------------------------------------
static int run_headless(Lenia& lenia, int steps,
                        const std::string& mass_log_path,
                        const std::string& snapshot_path) {
    // Open mass log file if requested
    std::ofstream mass_log;
    if (!mass_log_path.empty()) {
        mass_log.open(mass_log_path);
        if (!mass_log.is_open()) {
            std::cerr << "Error: cannot open mass log '" << mass_log_path << "'\n";
            return 1;
        }
        mass_log << "step,mass\n";
        mass_log << 0 << "," << lenia.total_mass() << "\n";
    }

    std::cout << "Running " << steps << " headless steps...\n";
    for (int s = 1; s <= steps; ++s) {
        lenia.step();
        if (mass_log.is_open()) {
            mass_log << s << "," << lenia.total_mass() << "\n";
        }
        if (s % 100 == 0 || s == steps) {
            std::cout << "  step " << s << "/" << steps
                      << "  mass=" << lenia.total_mass() << "\n";
        }
    }

    // Save snapshot if requested
    if (!snapshot_path.empty() && lenia.dimension() == 3) {
        save_snapshot(lenia.grid_3d(), lenia.grid_size(), snapshot_path);
    }

    std::cout << "Headless run complete.\n";
    return 0;
}

int main(int argc, char* argv[]) {
    LeniaConfig config = parse_args(argc, argv);
    print_config(config);

    std::string pattern = extract_flag(argc, argv, "--pattern", "orbium");
    bool headless = has_flag(argc, argv, "--headless");

    Lenia lenia(config);

    // --- Initialize pattern ---
    std::string cells_path = extract_flag(argc, argv, "--cells", "");

    if (!cells_path.empty() && config.dimension == 3) {
        if (!load_cells_into_grid(cells_path, lenia.grid_3d_mut(), config.grid_size))
            return 1;
    } else if (pattern == "random") {
        lenia.init_random(config.seed);
    } else if (pattern == "orbium" && config.dimension == 2) {
        lenia.init_orbium();
    } else if (pattern == "geminium" && config.dimension == 2) {
        lenia.init_geminium();
    } else if (pattern == "ring" && config.dimension == 2) {
        lenia.init_2d_ring();
    } else if (pattern == "multi" && config.dimension == 2) {
        lenia.init_2d_multi();
    } else if (pattern == "glider" && config.dimension == 3) {
        lenia.init_3d_glider();
    } else if (pattern == "multi" && config.dimension == 3) {
        lenia.init_3d_multi();
    } else if (pattern == "shell" && config.dimension == 3) {
        lenia.init_3d_shell();
    } else if (pattern == "dipole" && config.dimension == 3) {
        lenia.init_3d_dipole();
    } else if (pattern == "blob" || (pattern == "orbium" && config.dimension == 3)) {
        lenia.init_blob();
    }

    // --- Headless mode ---
    if (headless) {
        int steps = std::atoi(extract_flag(argc, argv, "--steps", "1000").c_str());
        std::string mass_log_path = extract_flag(argc, argv, "--mass-log", "");
        std::string snapshot_path = extract_flag(argc, argv, "--snapshot", "");
        return run_headless(lenia, steps, mass_log_path, snapshot_path);
    }

    // --- Interactive mode ---
    RenderConfig render_config;
    run_visualization(lenia, render_config);

    return 0;
}
