#pragma once

#include "grid.h"
#include "fft_engine.h"
#include "kernel.h"
#include "growth.h"

#include <memory>
#include <vector>
#include <complex>

// Pairs a kernel with its growth function parameters (for multi-kernel Lenia)
struct KernelGrowthPair {
    KernelParams kernel;
    GrowthParams growth;
};

struct LeniaConfig {
    int dimension   = 2;     // 2 or 3
    int grid_size   = 256;   // N for NxN or NxNxN
    KernelParams kernel;     // Primary kernel (backward compatible)
    GrowthParams growth;     // Primary growth params
    std::vector<KernelGrowthPair> extra_kernels;  // Additional kernel-growth pairs
    int T           = 10;    // Time resolution (dt = 1/T)
    int num_threads = 1;
    unsigned seed   = 42;

    int num_kernels() const { return 1 + static_cast<int>(extra_kernels.size()); }
};

class Lenia {
public:
    explicit Lenia(const LeniaConfig& config);

    void step();
    int iteration() const { return iteration_; }
    double total_mass() const;
    float dt() const { return dt_; }
    const LeniaConfig& config() const { return config_; }
    int num_kernels() const { return config_.num_kernels(); }

    // Access to grid state
    const float* state_data() const;
    int grid_size() const { return config_.grid_size; }
    int dimension() const { return config_.dimension; }

    // Access to last growth field G(U) — available after each step().
    // Values in [-1, +1]: positive = growth, negative = decay.
    const float* growth_data() const { return growth_buffer_.data(); }

    // For 2D
    const Grid<2>& grid_2d() const;
    Grid<2>& grid_2d_mut();

    // For 3D
    const Grid<3>& grid_3d() const;
    Grid<3>& grid_3d_mut();

    // Initialize with specific pattern
    void init_random(unsigned seed);
    void init_orbium();      // Known 2D glider pattern (Chan 2019)
    void init_geminium();    // Self-replicating 2D species (Chan 2019)
    void init_2d_ring();     // Ring pattern — pulsing/splitting behavior
    void init_2d_multi();    // Multiple Orbium interacting
    void init_blob();        // Gaussian blob at center (works for 2D and 3D)

    // 3D-specific patterns for pattern discovery
    void init_3d_glider();   // Asymmetric blob that tends to glide
    void init_3d_multi();    // Multiple interacting blobs
    void init_3d_shell();    // Hollow spherical shell
    void init_3d_dipole();   // Two close blobs of different intensity

    // Update parameters at runtime
    void set_growth_params(const GrowthParams& params);
    void set_kernel_params(const KernelParams& params);  // Recomputes kernel FFT
    void set_time_step(int T);

private:
    void step_2d();
    void step_3d();
    void rebuild_kernel();

    LeniaConfig config_;
    float dt_;
    int iteration_ = 0;

    // 2D state
    std::unique_ptr<Grid<2>> grid_2d_;
    std::unique_ptr<FFTEngine> fft_2d_;
    std::vector<std::complex<float>> kernel_spectrum_2d_;

    // 3D state
    std::unique_ptr<Grid<3>> grid_3d_;
    std::unique_ptr<FFTEngine> fft_3d_;
    std::vector<std::complex<float>> kernel_spectrum_3d_;

    // Temporary buffer for growth function output
    std::vector<float> growth_buffer_;

    // Multi-kernel support: spectra for each additional kernel
    std::vector<std::vector<std::complex<float>>> extra_spectra_2d_;
    std::vector<std::vector<std::complex<float>>> extra_spectra_3d_;

    // Temporary buffers for multi-kernel step
    std::vector<float> conv_buffer_;                 // single-kernel convolution result
    std::vector<std::complex<float>> saved_spectrum_; // saved state FFT
};
