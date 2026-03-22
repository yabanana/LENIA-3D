// main_sweep.cc
//
// Headless parameter sweep: runs multiple sigma values for a 3D pattern
// and prints mass over time to identify the stability zone.
// Usage: ./lenia_sweep --size 128 --threads 8 --pattern shell

#include "core/config.h"
#include "core/lenia.h"

#include <cstdio>
#include <cstring>
#include <string>

static std::string extract_pattern(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--pattern") == 0 && i + 1 < argc)
            return argv[i + 1];
    }
    return "blob";
}

static void init_pattern(Lenia& lenia, const std::string& pat) {
    if (pat == "glider")     lenia.init_3d_glider();
    else if (pat == "multi") lenia.init_3d_multi();
    else if (pat == "shell") lenia.init_3d_shell();
    else if (pat == "dipole")lenia.init_3d_dipole();
    else                     lenia.init_blob();
}

int main(int argc, char* argv[]) {
    LeniaConfig base = parse_args(argc, argv);
    base.dimension = 3;
    if (base.grid_size > 128) base.grid_size = 128; // keep sweep fast

    std::string pattern = extract_pattern(argc, argv);

    // Sweep sigma from 0.015 to 0.035 in steps of 0.001
    const float sigma_min = 0.015f;
    const float sigma_max = 0.035f;
    const float sigma_step = 0.001f;
    const int steps = 200;

    std::printf("# Pattern: %s  Size: %d  R: %d  mu: %.3f  T: %d  threads: %d\n",
                pattern.c_str(), base.grid_size, base.kernel.radius,
                base.growth.mu, base.T, base.num_threads);
    std::printf("# sigma, step, mass\n");

    for (float sigma = sigma_min; sigma <= sigma_max + 1e-6f; sigma += sigma_step) {
        LeniaConfig config = base;
        config.growth.sigma = sigma;

        Lenia lenia(config);
        init_pattern(lenia, pattern);

        double initial_mass = lenia.total_mass();

        // Run and sample mass at key points
        for (int s = 1; s <= steps; ++s) {
            lenia.step();
            if (s == 50 || s == 100 || s == 150 || s == 200) {
                double mass = lenia.total_mass();
                double ratio = (initial_mass > 0) ? mass / initial_mass : 0;
                std::printf("%.4f, %3d, %.1f, ratio=%.3f\n",
                            sigma, s, mass, ratio);
            }
        }
    }

    return 0;
}
