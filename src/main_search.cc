// main_search.cc
//
// Automated 3D Lenia parameter-space search tool.
// Sweeps over (sigma, mu, radius, T, pattern) combinations, computes
// scientific metrics at each sample point, classifies the run, and
// outputs results as CSV.
//
// Usage: ./lenia_search [options]
//   --sigma-min F       (default: 0.010)
//   --sigma-max F       (default: 0.050)
//   --sigma-step F      (default: 0.002)
//   --mu-min F          (default: 0.10)
//   --mu-max F          (default: 0.20)
//   --mu-step F         (default: 0.02)
//   --radii LIST        comma-separated (default: 8,10,13,16)
//   --T-values LIST     comma-separated (default: 10,20,30)
//   --patterns LIST     comma-separated (default: blob,shell,dipole,multi,glider)
//   --size N            grid side length (default: 96)
//   --steps N           simulation steps (default: 200)
//   --sample-every N    sample interval (default: 10)
//   --threads N         OpenMP threads (default: 1)
//   --per-step          output per-step CSV (verbose)
//   --no-early-term     disable early termination
//   --help              print usage

#include "core/lenia.h"
#include "search/metrics_3d.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <cmath>

// ---------------------------------------------------------------------------
// CLI parsing helpers
// ---------------------------------------------------------------------------
static void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --sigma-min F       (default: 0.010)\n"
        "  --sigma-max F       (default: 0.050)\n"
        "  --sigma-step F      (default: 0.002)\n"
        "  --mu-min F          (default: 0.10)\n"
        "  --mu-max F          (default: 0.20)\n"
        "  --mu-step F         (default: 0.02)\n"
        "  --radii LIST        comma-separated (default: 8,10,13,16)\n"
        "  --T-values LIST     comma-separated (default: 10,20,30)\n"
        "  --patterns LIST     comma-separated (default: blob,shell,dipole,multi,glider)\n"
        "  --size N            grid side length (default: 96)\n"
        "  --steps N           simulation steps (default: 200)\n"
        "  --sample-every N    sample interval (default: 10)\n"
        "  --threads N         OpenMP threads (default: 1)\n"
        "  --per-step          output per-step CSV (verbose)\n"
        "  --no-early-term     disable early termination\n"
        "  --help              print this message\n",
        argv0);
}

static std::vector<int> parse_int_list(const char* s) {
    std::vector<int> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        result.push_back(std::atoi(token.c_str()));
    }
    return result;
}

static std::vector<std::string> parse_string_list(const char* s) {
    std::vector<std::string> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        result.push_back(token);
    }
    return result;
}

struct SearchConfig {
    float sigma_min   = 0.010f;
    float sigma_max   = 0.050f;
    float sigma_step  = 0.002f;
    float mu_min      = 0.10f;
    float mu_max      = 0.20f;
    float mu_step     = 0.02f;
    std::vector<int> radii       = {8, 10, 13, 16};
    std::vector<int> T_values    = {10, 20, 30};
    std::vector<std::string> patterns = {"blob", "shell", "dipole", "multi", "glider"};
    int size          = 96;
    int steps         = 200;
    int sample_every  = 10;
    int threads       = 1;
    bool per_step     = false;
    bool early_term   = true;
};

static SearchConfig parse_search_args(int argc, char* argv[]) {
    SearchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        auto match = [&](const char* flag) {
            return std::strcmp(argv[i], flag) == 0;
        };
        auto next_f = [&]() -> float {
            return (i + 1 < argc) ? std::strtof(argv[++i], nullptr) : 0.0f;
        };
        auto next_i = [&]() -> int {
            return (i + 1 < argc) ? std::atoi(argv[++i]) : 0;
        };
        auto next_s = [&]() -> const char* {
            return (i + 1 < argc) ? argv[++i] : "";
        };

        if (match("--help"))           { print_usage(argv[0]); std::exit(0); }
        else if (match("--sigma-min")) { cfg.sigma_min = next_f(); }
        else if (match("--sigma-max")) { cfg.sigma_max = next_f(); }
        else if (match("--sigma-step")){ cfg.sigma_step = next_f(); }
        else if (match("--mu-min"))    { cfg.mu_min = next_f(); }
        else if (match("--mu-max"))    { cfg.mu_max = next_f(); }
        else if (match("--mu-step"))   { cfg.mu_step = next_f(); }
        else if (match("--radii"))     { cfg.radii = parse_int_list(next_s()); }
        else if (match("--T-values"))  { cfg.T_values = parse_int_list(next_s()); }
        else if (match("--patterns"))  { cfg.patterns = parse_string_list(next_s()); }
        else if (match("--size"))      { cfg.size = next_i(); }
        else if (match("--steps"))     { cfg.steps = next_i(); }
        else if (match("--sample-every")) { cfg.sample_every = next_i(); }
        else if (match("--threads"))   { cfg.threads = next_i(); }
        else if (match("--per-step"))  { cfg.per_step = true; }
        else if (match("--no-early-term")) { cfg.early_term = false; }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Pattern initialization dispatch (reuses Lenia init methods)
// ---------------------------------------------------------------------------
static void init_pattern(Lenia& lenia, const std::string& pat) {
    if (pat == "glider")      lenia.init_3d_glider();
    else if (pat == "multi")  lenia.init_3d_multi();
    else if (pat == "shell")  lenia.init_3d_shell();
    else if (pat == "dipole") lenia.init_3d_dipole();
    else                      lenia.init_blob();
}

// ---------------------------------------------------------------------------
// Count total runs for progress reporting
// ---------------------------------------------------------------------------
static int count_range(float min_val, float max_val, float step) {
    if (step <= 0.0f) return 1;
    return static_cast<int>((max_val - min_val) / step + 1.5f);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    SearchConfig cfg = parse_search_args(argc, argv);

    int n_sigma = count_range(cfg.sigma_min, cfg.sigma_max, cfg.sigma_step);
    int n_mu    = count_range(cfg.mu_min, cfg.mu_max, cfg.mu_step);
    int total_runs = static_cast<int>(cfg.radii.size())
                   * static_cast<int>(cfg.T_values.size())
                   * static_cast<int>(cfg.patterns.size())
                   * n_mu * n_sigma;

    // Print config as comment header
    std::printf("# lenia_search config\n");
    std::printf("# size=%d steps=%d sample_every=%d threads=%d early_term=%d\n",
                cfg.size, cfg.steps, cfg.sample_every, cfg.threads,
                cfg.early_term ? 1 : 0);
    std::printf("# sigma=[%.4f:%.4f:%.4f] mu=[%.4f:%.4f:%.4f]\n",
                cfg.sigma_min, cfg.sigma_max, cfg.sigma_step,
                cfg.mu_min, cfg.mu_max, cfg.mu_step);
    std::printf("# radii=");
    for (std::size_t i = 0; i < cfg.radii.size(); ++i)
        std::printf("%s%d", i ? "," : "", cfg.radii[i]);
    std::printf(" T_values=");
    for (std::size_t i = 0; i < cfg.T_values.size(); ++i)
        std::printf("%s%d", i ? "," : "", cfg.T_values[i]);
    std::printf(" patterns=");
    for (std::size_t i = 0; i < cfg.patterns.size(); ++i)
        std::printf("%s%s", i ? "," : "", cfg.patterns[i].c_str());
    std::printf("\n");
    std::printf("# total_runs=%d\n", total_runs);

    // Print per-step header if verbose mode
    if (cfg.per_step) {
        std::printf("run_id,sigma,mu,radius,T,pattern,step,mass,mass_ratio,"
                    "com_x,com_y,com_z,com_disp,rg,compactness,entropy,sv_ratio\n");
    } else {
        std::printf("run_id,sigma,mu,radius,T,pattern,steps_run,early_term,class,"
                    "final_mass_ratio,mass_variance,mean_com_speed,"
                    "final_rg,final_compactness,final_entropy,final_sv_ratio,"
                    "dom_freq,osc_amp,bbox_dx,bbox_dy,bbox_dz\n");
    }

    int run_id = 0;
    auto t_start = std::chrono::steady_clock::now();

    // Outer loop: R and T (kernel FFT computed once per (R, T) combination)
    for (int R : cfg.radii) {
        for (int T_val : cfg.T_values) {
            // Build Lenia with this (R, T) — kernel FFT computed once
            LeniaConfig lenia_cfg;
            lenia_cfg.dimension = 3;
            lenia_cfg.grid_size = cfg.size;
            lenia_cfg.kernel.radius = R;
            lenia_cfg.T = T_val;
            lenia_cfg.num_threads = cfg.threads;
            lenia_cfg.growth.mu = cfg.mu_min;
            lenia_cfg.growth.sigma = cfg.sigma_min;

            Lenia lenia(lenia_cfg);

            for (const auto& pat : cfg.patterns) {
                for (float mu = cfg.mu_min;
                     mu <= cfg.mu_max + 1e-6f;
                     mu += cfg.mu_step)
                {
                    for (float sigma = cfg.sigma_min;
                         sigma <= cfg.sigma_max + 1e-6f;
                         sigma += cfg.sigma_step)
                    {
                        ++run_id;

                        // Set growth params without rebuilding kernel
                        GrowthParams gp;
                        gp.mu = mu;
                        gp.sigma = sigma;
                        lenia.set_growth_params(gp);

                        // Reset grid with pattern
                        init_pattern(lenia, pat);

                        // Compute initial metrics
                        double initial_mass = lenia.total_mass();
                        auto initial_com = compute_center_of_mass(lenia.grid_3d());

                        std::vector<MetricsSnapshot> history;
                        history.reserve(cfg.steps / cfg.sample_every + 1);

                        bool early_stopped = false;
                        int steps_done = 0;

                        for (int s = 1; s <= cfg.steps; ++s) {
                            lenia.step();
                            steps_done = s;

                            if (s % cfg.sample_every == 0 || s == cfg.steps) {
                                auto snap = compute_all_metrics(
                                    lenia.grid_3d(), s, initial_mass,
                                    initial_com);
                                history.push_back(snap);

                                // Per-step output
                                if (cfg.per_step) {
                                    std::printf("%d,%.4f,%.4f,%d,%d,%s,%d,"
                                        "%.2f,%.4f,%.2f,%.2f,%.2f,%.2f,"
                                        "%.2f,%.6f,%.4f,%.4f\n",
                                        run_id, sigma, mu, R, T_val,
                                        pat.c_str(), s,
                                        snap.mass, snap.mass_ratio,
                                        snap.com[0], snap.com[1], snap.com[2],
                                        snap.com_displacement,
                                        snap.radius_of_gyration,
                                        snap.compactness,
                                        snap.spatial_entropy,
                                        snap.surface_volume_ratio);
                                }

                                // Early termination check at step 50
                                if (cfg.early_term && s >= 50 &&
                                    history.size() >= 2) {
                                    double mr = snap.mass_ratio;
                                    if (mr < 0.05) {
                                        early_stopped = true;
                                        break;
                                    }
                                    if (mr > 0.80 && snap.compactness < 0.1) {
                                        early_stopped = true;
                                        break;
                                    }
                                }
                            }
                        }

                        // Summary output
                        if (!cfg.per_step) {
                            auto summary = classify_run(
                                history, sigma, mu, R, T_val, pat,
                                early_stopped, steps_done);

                            std::printf("%d,%.4f,%.4f,%d,%d,%s,%d,%d,%s,"
                                "%.6f,%.2e,%.4f,"
                                "%.2f,%.6f,%.4f,%.4f,"
                                "%.4f,%.4f,%d,%d,%d\n",
                                run_id, sigma, mu, R, T_val, pat.c_str(),
                                summary.steps_run,
                                summary.early_terminated ? 1 : 0,
                                pattern_class_name(summary.classification),
                                summary.final_mass_ratio,
                                summary.mass_variance,
                                summary.mean_com_speed,
                                summary.final_rg,
                                summary.final_compactness,
                                summary.final_entropy,
                                summary.final_sv_ratio,
                                summary.dominant_frequency,
                                summary.oscillation_amplitude,
                                summary.final_bbox_size[0],
                                summary.final_bbox_size[1],
                                summary.final_bbox_size[2]);
                        }

                        // Progress on stderr
                        if (run_id % 10 == 0 || run_id == total_runs) {
                            auto now = std::chrono::steady_clock::now();
                            double elapsed = std::chrono::duration<double>(
                                now - t_start).count();
                            double per_run = elapsed / run_id;
                            double remaining = per_run * (total_runs - run_id);
                            std::fprintf(stderr,
                                "\r[%d/%d] %.0f%% | %.1fs elapsed | "
                                "~%.0fs remaining  ",
                                run_id, total_runs,
                                100.0 * run_id / total_runs,
                                elapsed, remaining);
                        }
                    }
                }
            }
        }
    }

    std::fprintf(stderr, "\nDone. %d runs completed.\n", run_id);
    return 0;
}
