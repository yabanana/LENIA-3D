#include "lenia.h"

#include <cassert>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
Lenia::Lenia(const LeniaConfig& config) : config_(config) {
    assert(config.dimension == 2 || config.dimension == 3);
    assert(config.grid_size > 0);
    assert(config.T > 0);

    dt_ = 1.0f / static_cast<float>(config.T);

    #ifdef _OPENMP
    omp_set_num_threads(config.num_threads);
    #endif

    int N = config.grid_size;

    if (config.dimension == 2) {
        grid_2d_ = std::make_unique<Grid<2>>(std::array<int, 2>{N, N});
        grid_2d_->fill(0.0f);

        int shape[2] = {N, N};
        fft_2d_ = std::make_unique<FFTEngine>(2, shape, config.num_threads);

        // Generate primary kernel and precompute its spectrum
        Grid<2> kern = generate_kernel_2d(N, N, config.kernel);
        kernel_spectrum_2d_ = precompute_kernel_spectrum(kern, config.num_threads);

        // Build extra kernel spectra
        for (const auto& kgp : config.extra_kernels) {
            Grid<2> ek = generate_kernel_2d(N, N, kgp.kernel);
            extra_spectra_2d_.push_back(
                precompute_kernel_spectrum(ek, config.num_threads));
        }

        std::size_t grid_n = static_cast<std::size_t>(N) * N;
        growth_buffer_.resize(grid_n, 0.0f);

        // Allocate multi-kernel buffers if needed
        if (config.num_kernels() > 1) {
            conv_buffer_.resize(grid_n, 0.0f);
            saved_spectrum_.resize(fft_2d_->spectrum_size());
        }

    } else {
        grid_3d_ = std::make_unique<Grid<3>>(std::array<int, 3>{N, N, N});
        grid_3d_->fill(0.0f);

        int shape[3] = {N, N, N};
        fft_3d_ = std::make_unique<FFTEngine>(3, shape, config.num_threads);

        Grid<3> kern = generate_kernel_3d(N, N, N, config.kernel);
        kernel_spectrum_3d_ = precompute_kernel_spectrum(kern, config.num_threads);

        // Build extra kernel spectra
        for (const auto& kgp : config.extra_kernels) {
            Grid<3> ek = generate_kernel_3d(N, N, N, kgp.kernel);
            extra_spectra_3d_.push_back(
                precompute_kernel_spectrum(ek, config.num_threads));
        }

        std::size_t grid_n = static_cast<std::size_t>(N) * N * N;
        growth_buffer_.resize(grid_n, 0.0f);

        // Allocate multi-kernel buffers if needed
        if (config.num_kernels() > 1) {
            conv_buffer_.resize(grid_n, 0.0f);
            saved_spectrum_.resize(fft_3d_->spectrum_size());
        }
    }
}

// ---------------------------------------------------------------------------
// step() -- dispatch to 2D or 3D
// ---------------------------------------------------------------------------
void Lenia::step() {
    if (config_.dimension == 2) {
        step_2d();
    } else {
        step_3d();
    }
    ++iteration_;
}

// ---------------------------------------------------------------------------
// step_2d() -- full Lenia pipeline for 2D
//   1. Forward FFT of current grid state
//   2. Pointwise multiply with precomputed kernel spectrum
//   3. Inverse FFT to get convolution result U (neighborhood potential)
//   4. Apply growth function G(U) with OpenMP
//   5. Update state: A = clip(A + dt * G(U), 0, 1)
// ---------------------------------------------------------------------------
void Lenia::step_2d() {
    assert(grid_2d_ && fft_2d_);

    float* state = grid_2d_->data();
    std::size_t n = grid_2d_->total_size();
    int nk = config_.num_kernels();

    if (nk == 1) {
        // --- Single kernel (fast path, unchanged) ---
        fft_2d_->forward(state);
        fft_2d_->pointwise_multiply(kernel_spectrum_2d_);
        fft_2d_->inverse(growth_buffer_.data());
        growth_batch(growth_buffer_.data(), growth_buffer_.data(), n,
                     config_.growth);
    } else {
        // --- Multi-kernel path ---
        // 1. Forward FFT of state and save spectrum
        fft_2d_->forward(state);
        fft_2d_->save_spectrum(saved_spectrum_);

        // Zero growth accumulator
        std::fill(growth_buffer_.begin(), growth_buffer_.end(), 0.0f);

        // 2. Kernel 0 (primary)
        fft_2d_->pointwise_multiply(kernel_spectrum_2d_);
        fft_2d_->inverse(conv_buffer_.data());
        growth_batch(conv_buffer_.data(), conv_buffer_.data(), n,
                     config_.growth);
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            growth_buffer_[i] += conv_buffer_[i];
        }

        // 3. Extra kernels
        for (std::size_t k = 0; k < extra_spectra_2d_.size(); ++k) {
            fft_2d_->load_spectrum(saved_spectrum_);
            fft_2d_->pointwise_multiply(extra_spectra_2d_[k]);
            fft_2d_->inverse(conv_buffer_.data());
            growth_batch(conv_buffer_.data(), conv_buffer_.data(), n,
                         config_.extra_kernels[k].growth);
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < n; ++i) {
                growth_buffer_[i] += conv_buffer_[i];
            }
        }

        // 4. Average growth across kernels
        float inv_nk = 1.0f / static_cast<float>(nk);
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            growth_buffer_[i] *= inv_nk;
        }
    }

    // Update state: A = clip(A + dt * G_total, 0, 1)
    float local_dt = dt_;
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        float val = state[i] + local_dt * growth_buffer_[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        state[i] = val;
    }
}

// ---------------------------------------------------------------------------
// step_3d() -- full Lenia pipeline for 3D
// ---------------------------------------------------------------------------
void Lenia::step_3d() {
    assert(grid_3d_ && fft_3d_);

    float* state = grid_3d_->data();
    std::size_t n = grid_3d_->total_size();
    int nk = config_.num_kernels();

    if (nk == 1) {
        // --- Single kernel (fast path, unchanged) ---
        fft_3d_->forward(state);
        fft_3d_->pointwise_multiply(kernel_spectrum_3d_);
        fft_3d_->inverse(growth_buffer_.data());
        growth_batch(growth_buffer_.data(), growth_buffer_.data(), n,
                     config_.growth);
    } else {
        // --- Multi-kernel path ---
        fft_3d_->forward(state);
        fft_3d_->save_spectrum(saved_spectrum_);

        std::fill(growth_buffer_.begin(), growth_buffer_.end(), 0.0f);

        // Kernel 0 (primary)
        fft_3d_->pointwise_multiply(kernel_spectrum_3d_);
        fft_3d_->inverse(conv_buffer_.data());
        growth_batch(conv_buffer_.data(), conv_buffer_.data(), n,
                     config_.growth);
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            growth_buffer_[i] += conv_buffer_[i];
        }

        // Extra kernels
        for (std::size_t k = 0; k < extra_spectra_3d_.size(); ++k) {
            fft_3d_->load_spectrum(saved_spectrum_);
            fft_3d_->pointwise_multiply(extra_spectra_3d_[k]);
            fft_3d_->inverse(conv_buffer_.data());
            growth_batch(conv_buffer_.data(), conv_buffer_.data(), n,
                         config_.extra_kernels[k].growth);
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < n; ++i) {
                growth_buffer_[i] += conv_buffer_[i];
            }
        }

        // Average growth across kernels
        float inv_nk = 1.0f / static_cast<float>(nk);
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            growth_buffer_[i] *= inv_nk;
        }
    }

    // Update state: A = clip(A + dt * G_total, 0, 1)
    float local_dt = dt_;
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        float val = state[i] + local_dt * growth_buffer_[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        state[i] = val;
    }
}

// ---------------------------------------------------------------------------
// total_mass()
// ---------------------------------------------------------------------------
double Lenia::total_mass() const {
    if (config_.dimension == 2) {
        assert(grid_2d_);
        return grid_2d_->total_mass();
    } else {
        assert(grid_3d_);
        return grid_3d_->total_mass();
    }
}

// ---------------------------------------------------------------------------
// state_data()
// ---------------------------------------------------------------------------
const float* Lenia::state_data() const {
    if (config_.dimension == 2) {
        assert(grid_2d_);
        return grid_2d_->data();
    } else {
        assert(grid_3d_);
        return grid_3d_->data();
    }
}

// ---------------------------------------------------------------------------
// Grid accessors
// ---------------------------------------------------------------------------
const Grid<2>& Lenia::grid_2d() const {
    assert(grid_2d_);
    return *grid_2d_;
}

Grid<2>& Lenia::grid_2d_mut() {
    assert(grid_2d_);
    return *grid_2d_;
}

const Grid<3>& Lenia::grid_3d() const {
    assert(grid_3d_);
    return *grid_3d_;
}

Grid<3>& Lenia::grid_3d_mut() {
    assert(grid_3d_);
    return *grid_3d_;
}

// ---------------------------------------------------------------------------
// init_random() -- fill with random values in [0, 1]
// ---------------------------------------------------------------------------
void Lenia::init_random(unsigned seed) {
    if (config_.dimension == 2) {
        grid_2d_->fill_random(0.0f, 1.0f, seed);
    } else {
        grid_3d_->fill_random(0.0f, 1.0f, seed);
    }
    iteration_ = 0;
}

// ---------------------------------------------------------------------------
// init_blob() -- Compact seed at grid center
//
// Creates a dense spherical seed with shell_r = 0.2*R, thickness = 0.2*R.
// This compact size places mass where the kernel has peak sensitivity (~R/2),
// producing convolution values U ≈ µ for typical growth parameters.
// In 2D, falls back to a Gaussian blob scaled to kernel radius.
// ---------------------------------------------------------------------------
void Lenia::init_blob() {
    const int N = config_.grid_size;
    const float center = static_cast<float>(N) / 2.0f;
    const float kernel_R = static_cast<float>(config_.kernel.radius);

    if (config_.dimension == 2) {
        grid_2d_->fill(0.0f);
        const float blob_r = kernel_R * 1.0f;
        const float sigma = blob_r / 3.0f;
        const float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float di = static_cast<float>(i) - center;
                float dj = static_cast<float>(j) - center;
                float r2 = di * di + dj * dj;
                float val = std::exp(-r2 * inv_2sigma2);
                if (val > 0.001f) {
                    grid_2d_->at(i, j) = val;
                }
            }
        }
    } else {
        grid_3d_->fill(0.0f);
        // Gaussian blob with sigma adapted so convolution U ≈ mu at center.
        // The formula sigma = R * (0.19 + 0.74*mu) was derived empirically
        // to produce initial growth G > 0 for both single-kernel (mu~0.15)
        // and multi-kernel 312 presets (mu~0.18–0.28).
        const float mu = config_.growth.mu;
        const float sigma = kernel_R * (0.19f + 0.74f * mu);
        const float inv_2s2 = 1.0f / (2.0f * sigma * sigma);
        const float cutoff_r = sigma * 3.0f;  // negligible beyond 3*sigma
        const int r_max = std::min(static_cast<int>(cutoff_r) + 1, N / 2);
        for (int k = -r_max; k <= r_max; ++k) {
            for (int i = -r_max; i <= r_max; ++i) {
                for (int j = -r_max; j <= r_max; ++j) {
                    float r2 = static_cast<float>(k*k + i*i + j*j);
                    float val = 0.85f * std::exp(-r2 * inv_2s2);
                    if (val < 0.01f) continue;
                    int gk = (static_cast<int>(center) + k % N + N) % N;
                    int gi = (static_cast<int>(center) + i % N + N) % N;
                    int gj = (static_cast<int>(center) + j % N + N) % N;
                    grid_3d_->at(gk, gi, gj) = val;
                }
            }
        }
    }
    iteration_ = 0;
}

// ---------------------------------------------------------------------------
// init_orbium() -- initialize with the Orbium glider pattern (Chan 2019)
//
// Orbium is a glider-like creature in Lenia discovered by Bert Chan.
// Parameters: R=13, T=10, mu=0.15, sigma=0.015, beta=[1],
//             kernel_core = (4*r*(1-r))^4
//
// The initial state is a 20x20 pattern placed at the grid center.
// The values below are an approximation of the canonical Orbium pattern
// from Chan 2019 "Lenia - Biology of Artificial Life", discretized onto
// a 20x20 patch. The pattern is a crescent-shaped organism that glides
// across the grid indefinitely with appropriate parameters.
// ---------------------------------------------------------------------------
void Lenia::init_orbium() {
    if (config_.dimension != 2) {
        throw std::runtime_error(
            "init_orbium() is only available for 2D simulations");
    }

    grid_2d_->fill(0.0f);

    // Canonical Orbium pattern decoded from Chan's animals.json (RLE encoding).
    // Bert Chan, "Lenia - Biology of Artificial Life", Complex Systems 28(3),
    // 2019. Orbium unicaudatus — a crescent-shaped glider with parameters
    // R=13, T=10, mu=0.15, sigma=0.015, beta=[1], kernel_core=(4r(1-r))^4.
    // Values exceed 1.0 in the initial state; they are clipped after step 1.
    // clang-format off
    static const float orbium_pattern[20][20] = {
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0510f, 0.0157f, 0.0000f,
         0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.3294f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2314f, 0.3020f, 0.3059f, 0.0824f,
         0.0706f, 0.1922f, 0.1961f, 0.0706f, 0.0039f, 0.3490f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0863f, 0.3608f, 0.4510f, 0.4784f, 0.4353f,
         0.2824f, 0.2784f, 0.2667f, 0.2784f, 0.2706f, 0.2000f, 0.4431f, 0.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0118f, 0.0667f, 0.4431f, 0.5490f, 0.5608f, 0.4745f,
         0.2235f, 0.0784f, 0.0549f, 0.0627f, 0.2157f, 0.3098f, 0.8000f, 0.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0353f, 0.2235f, 0.2784f, 0.4353f, 0.4980f, 0.4784f, 0.3765f,
         0.2275f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0471f, 0.4941f, 0.4235f, 0.0000f, 0.0000f},
        {0.0039f, 0.0000f, 0.0157f, 0.2314f, 0.2627f, 0.2275f, 0.2039f, 0.3412f, 0.3647f, 0.3569f,
         0.3020f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0196f, 0.9569f, 0.0000f, 0.0000f},
        {0.3294f, 0.0000f, 0.1961f, 0.2667f, 0.0784f, 0.0000f, 0.0000f, 0.2941f, 0.4039f, 0.4627f,
         0.4667f, 0.3333f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.5490f, 0.2510f, 0.0000f},
        {0.0000f, 0.2157f, 0.2784f, 0.2039f, 0.0000f, 0.0000f, 0.0000f, 0.3647f, 0.5216f, 0.6000f,
         0.6314f, 0.6039f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.6118f, 0.0000f},
        {0.0000f, 0.6824f, 0.3098f, 0.0314f, 0.0000f, 0.0000f, 0.0000f, 0.2824f, 0.6235f, 0.7294f,
         0.7765f, 0.7765f, 0.5137f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.5216f, 0.0000f},
        {0.0000f, 0.6471f, 0.3294f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0275f, 0.7137f, 0.8510f,
         0.9333f, 0.9490f, 0.9020f, 0.2000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.3765f, 0.1922f},
        {0.0000f, 0.0000f, 0.6902f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.7608f, 0.9686f,
         1.0549f, 1.0863f, 1.0667f, 0.6549f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.3176f, 0.2353f},
        {0.0000f, 0.0000f, 0.9333f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.6000f, 1.0627f,
         1.0941f, 1.0941f, 1.0941f, 1.0078f, 0.3725f, 0.0000f, 0.0000f, 0.0235f, 0.3137f, 0.2275f},
        {0.0000f, 0.0000f, 0.6471f, 0.0745f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.4275f, 1.0941f,
         1.0941f, 1.0745f, 1.0941f, 1.0667f, 0.6510f, 0.2510f, 0.0510f, 0.2118f, 0.3373f, 0.0863f},
        {0.0000f, 0.0000f, 0.0314f, 0.5412f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2706f, 0.9451f,
         1.0941f, 0.9882f, 0.9569f, 0.9608f, 0.7451f, 0.4275f, 0.2902f, 0.3098f, 0.3255f, 0.0392f},
        {0.0000f, 0.0000f, 0.0000f, 0.5176f, 0.2078f, 0.0000f, 0.0000f, 0.0000f, 0.2078f, 0.7137f,
         0.9725f, 0.9176f, 0.8745f, 0.8353f, 0.7059f, 0.5020f, 0.3882f, 0.3608f, 0.2588f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0784f, 0.4275f, 0.0745f, 0.0000f, 0.0000f, 0.2118f, 0.5176f,
         0.7686f, 0.8157f, 0.7725f, 0.7216f, 0.6196f, 0.4980f, 0.4039f, 0.3176f, 0.0627f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2588f, 0.3529f, 0.2431f, 0.2118f, 0.2667f, 0.4314f,
         0.5922f, 0.6510f, 0.6392f, 0.5922f, 0.5216f, 0.4314f, 0.3373f, 0.2118f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2392f, 0.3255f, 0.3294f, 0.3529f, 0.4118f,
         0.4824f, 0.5176f, 0.5059f, 0.4549f, 0.4000f, 0.3216f, 0.2196f, 0.0196f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0706f, 0.2627f, 0.3216f, 0.3451f,
         0.3686f, 0.3725f, 0.3529f, 0.3255f, 0.2588f, 0.0941f, 0.0196f, 0.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0588f, 0.1961f,
         0.2235f, 0.2275f, 0.2118f, 0.0784f, 0.0431f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
    };
    // clang-format on

    int N = config_.grid_size;
    int patch_size = 20;
    int offset_i = N / 2 - patch_size / 2;
    int offset_j = N / 2 - patch_size / 2;

    for (int pi = 0; pi < patch_size; ++pi) {
        for (int pj = 0; pj < patch_size; ++pj) {
            int gi = (offset_i + pi) % N;
            int gj = (offset_j + pj) % N;
            grid_2d_->at(gi, gj) = orbium_pattern[pi][pj];
        }
    }

    iteration_ = 0;
}

// ---------------------------------------------------------------------------
// init_geminium() -- Self-replicating 2D species (Chan 2019)
//
// Geminium natans — a two-lobed creature that periodically splits into two
// copies, each of which can split again. One of the most remarkable
// Lenia species. Best with R=10, T=10, mu=0.14, sigma=0.014.
// The initial pattern is a compact asymmetric two-lobed structure.
// ---------------------------------------------------------------------------
void Lenia::init_geminium() {
    if (config_.dimension != 2) {
        throw std::runtime_error("init_geminium() requires 2D");
    }
    grid_2d_->fill(0.0f);
    const int N = config_.grid_size;
    const float cx = static_cast<float>(N) / 2.0f;
    const float cy = static_cast<float>(N) / 2.0f;

    // Two-lobed initial structure: two offset gaussian blobs with a bridge
    auto gauss = [](float x, float y, float mx, float my, float sx, float sy) {
        return std::exp(-((x-mx)*(x-mx))/(2*sx*sx) - ((y-my)*(y-my))/(2*sy*sy));
    };

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float fi = static_cast<float>(i);
            float fj = static_cast<float>(j);
            // Left lobe
            float v1 = 0.9f * gauss(fi, fj, cy - 4.0f, cx - 3.0f, 3.5f, 3.0f);
            // Right lobe
            float v2 = 0.85f * gauss(fi, fj, cy + 3.0f, cx + 4.0f, 3.0f, 3.5f);
            // Bridge connecting them
            float v3 = 0.5f * gauss(fi, fj, cy, cx, 2.0f, 6.0f);
            float val = v1 + v2 + v3;
            if (val > 0.01f) {
                grid_2d_->at(i, j) = std::min(val, 1.0f);
            }
        }
    }
    iteration_ = 0;
}

// ---------------------------------------------------------------------------
// init_2d_ring() -- Ring pattern for pulsing/splitting dynamics
//
// A thin annular ring of active cells. Depending on parameters, this can
// pulse, contract, expand, or break into multiple solitons. Interesting
// with mu=0.15, sigma=0.02, R=13.
// ---------------------------------------------------------------------------
void Lenia::init_2d_ring() {
    if (config_.dimension != 2) {
        throw std::runtime_error("init_2d_ring() requires 2D");
    }
    grid_2d_->fill(0.0f);
    const int N = config_.grid_size;
    const float center = static_cast<float>(N) / 2.0f;
    const float R = static_cast<float>(config_.kernel.radius);
    const float ring_r = R * 0.8f;
    const float ring_w = R * 0.2f;
    const float inv_2w2 = 1.0f / (2.0f * ring_w * ring_w);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float di = static_cast<float>(i) - center;
            float dj = static_cast<float>(j) - center;
            float dist = std::sqrt(di * di + dj * dj);
            float diff = dist - ring_r;
            float val = 0.9f * std::exp(-diff * diff * inv_2w2);
            // Add slight asymmetry to break circular symmetry
            float angle = std::atan2(di, dj);
            val *= (1.0f + 0.15f * std::sin(3.0f * angle));
            if (val > 0.01f) {
                grid_2d_->at(i, j) = std::min(val, 1.0f);
            }
        }
    }
    iteration_ = 0;
}

// ---------------------------------------------------------------------------
// init_2d_multi() -- Multiple Orbium placed at different positions
//
// Four Orbium placed in a diamond arrangement. They interact when their
// neighborhoods overlap, creating complex dynamics: collision, scattering,
// merging, or annihilation depending on spacing and parameters.
// ---------------------------------------------------------------------------
void Lenia::init_2d_multi() {
    if (config_.dimension != 2) {
        throw std::runtime_error("init_2d_multi() requires 2D");
    }
    grid_2d_->fill(0.0f);
    const int N = config_.grid_size;
    const float R = static_cast<float>(config_.kernel.radius);

    // Reuse the Orbium pattern from init_orbium
    static const float orbium_pattern[20][20] = {
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0510f, 0.0157f, 0.0000f,
         0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.3294f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2314f, 0.3020f, 0.3059f, 0.0824f,
         0.0706f, 0.1922f, 0.1961f, 0.0706f, 0.0039f, 0.3490f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0863f, 0.3608f, 0.4510f, 0.4784f, 0.4353f,
         0.2824f, 0.2784f, 0.2667f, 0.2784f, 0.2706f, 0.2000f, 0.4431f, 0.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0118f, 0.0667f, 0.4431f, 0.5490f, 0.5608f, 0.4745f,
         0.2235f, 0.0784f, 0.0549f, 0.0627f, 0.2157f, 0.3098f, 0.8000f, 0.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0353f, 0.2235f, 0.2784f, 0.4353f, 0.4980f, 0.4784f, 0.3765f,
         0.2275f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0471f, 0.4941f, 0.4235f, 0.0000f, 0.0000f},
        {0.0039f, 0.0000f, 0.0157f, 0.2314f, 0.2627f, 0.2275f, 0.2039f, 0.3412f, 0.3647f, 0.3569f,
         0.3020f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0196f, 0.9569f, 0.0000f, 0.0000f},
        {0.3294f, 0.0000f, 0.1961f, 0.2667f, 0.0784f, 0.0000f, 0.0000f, 0.2941f, 0.4039f, 0.4627f,
         0.4667f, 0.3333f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.5490f, 0.2510f, 0.0000f},
        {0.0000f, 0.2157f, 0.2784f, 0.2039f, 0.0000f, 0.0000f, 0.0000f, 0.3647f, 0.5216f, 0.6000f,
         0.6314f, 0.6039f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.6118f, 0.0000f},
        {0.0000f, 0.6824f, 0.3098f, 0.0314f, 0.0000f, 0.0000f, 0.0000f, 0.2824f, 0.6235f, 0.7294f,
         0.7765f, 0.7765f, 0.5137f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.5216f, 0.0000f},
        {0.0000f, 0.6471f, 0.3294f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0275f, 0.7137f, 0.8510f,
         0.9333f, 0.9490f, 0.9020f, 0.2000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.3765f, 0.1922f},
        {0.0000f, 0.0000f, 0.6902f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.7608f, 0.9686f,
         1.0549f, 1.0863f, 1.0667f, 0.6549f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.3176f, 0.2353f},
        {0.0000f, 0.0000f, 0.9333f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.6000f, 1.0627f,
         1.0941f, 1.0941f, 1.0941f, 1.0078f, 0.3725f, 0.0000f, 0.0000f, 0.0235f, 0.3137f, 0.2275f},
        {0.0000f, 0.0000f, 0.6471f, 0.0745f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.4275f, 1.0941f,
         1.0941f, 1.0745f, 1.0941f, 1.0667f, 0.6510f, 0.2510f, 0.0510f, 0.2118f, 0.3373f, 0.0863f},
        {0.0000f, 0.0000f, 0.0314f, 0.5412f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2706f, 0.9451f,
         1.0941f, 0.9882f, 0.9569f, 0.9608f, 0.7451f, 0.4275f, 0.2902f, 0.3098f, 0.3255f, 0.0392f},
        {0.0000f, 0.0000f, 0.0000f, 0.5176f, 0.2078f, 0.0000f, 0.0000f, 0.0000f, 0.2078f, 0.7137f,
         0.9725f, 0.9176f, 0.8745f, 0.8353f, 0.7059f, 0.5020f, 0.3882f, 0.3608f, 0.2588f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0784f, 0.4275f, 0.0745f, 0.0000f, 0.0000f, 0.2118f, 0.5176f,
         0.7686f, 0.8157f, 0.7725f, 0.7216f, 0.6196f, 0.4980f, 0.4039f, 0.3176f, 0.0627f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2588f, 0.3529f, 0.2431f, 0.2118f, 0.2667f, 0.4314f,
         0.5922f, 0.6510f, 0.6392f, 0.5922f, 0.5216f, 0.4314f, 0.3373f, 0.2118f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.2392f, 0.3255f, 0.3294f, 0.3529f, 0.4118f,
         0.4824f, 0.5176f, 0.5059f, 0.4549f, 0.4000f, 0.3216f, 0.2196f, 0.0196f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0706f, 0.2627f, 0.3216f, 0.3451f,
         0.3686f, 0.3725f, 0.3529f, 0.3255f, 0.2588f, 0.0941f, 0.0196f, 0.0000f, 0.0000f, 0.0000f},
        {0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0588f, 0.1961f,
         0.2235f, 0.2275f, 0.2118f, 0.0784f, 0.0431f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f},
    };

    int patch_size = 20;
    float sep = R * 4.0f; // separation between creatures

    // Place 4 Orbium in a diamond, each rotated differently via placement offset
    int positions[4][2] = {
        {N/2 - (int)sep, N/2},
        {N/2 + (int)sep, N/2},
        {N/2, N/2 - (int)sep},
        {N/2, N/2 + (int)sep}
    };

    for (int p = 0; p < 4; ++p) {
        int oi = positions[p][0] - patch_size / 2;
        int oj = positions[p][1] - patch_size / 2;
        for (int pi = 0; pi < patch_size; ++pi) {
            for (int pj = 0; pj < patch_size; ++pj) {
                int gi = ((oi + pi) % N + N) % N;
                int gj = ((oj + pj) % N + N) % N;
                float val = grid_2d_->at(gi, gj) + orbium_pattern[pi][pj];
                grid_2d_->at(gi, gj) = std::min(val, 1.0f);
            }
        }
    }
    iteration_ = 0;
}

// ---------------------------------------------------------------------------
// Helper: place a compact 3D seed at (cx,cy,cz) with given kernel_R and peak.
// Uses shell_r = 0.2*kernel_R, thickness = 0.2*kernel_R (dense ball).
// ---------------------------------------------------------------------------
static void place_compact_seed(Grid<3>& grid, int N,
                               float cx, float cy, float cz,
                               float kernel_R, float peak) {
    float shell_r = kernel_R * 0.2f;
    float thickness = kernel_R * 0.2f;
    float inv_2t2 = 1.0f / (2.0f * thickness * thickness);
    int r_max = static_cast<int>(shell_r + 3.0f * thickness) + 1;

    int icx = static_cast<int>(cx);
    int icy = static_cast<int>(cy);
    int icz = static_cast<int>(cz);

    for (int dk = -r_max; dk <= r_max; ++dk) {
        for (int di = -r_max; di <= r_max; ++di) {
            for (int dj = -r_max; dj <= r_max; ++dj) {
                float dist = std::sqrt(static_cast<float>(dk*dk + di*di + dj*dj));
                float diff = dist - shell_r;
                float val = peak * std::exp(-diff * diff * inv_2t2);
                if (val < 0.01f) continue;

                int k = ((icz + dk) % N + N) % N;
                int i = ((icy + di) % N + N) % N;
                int j = ((icx + dj) % N + N) % N;

                float cur = grid.at(k, i, j);
                float sum = cur + val;
                grid.at(k, i, j) = (sum > 1.0f) ? 1.0f : sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// init_3d_glider() -- Asymmetric compact seed for glider-like behavior
//
// Two overlapping compact seeds offset along x with different intensities
// break symmetry to bias drift direction.  Under appropriate parameters,
// this can evolve into a traveling structure.
//
// Note: P2/P6 (µ > 0.150) sit at the edge of the survival zone for compact
// seeds; these parameters may require fine-tuned initial conditions.
// ---------------------------------------------------------------------------
void Lenia::init_3d_glider() {
    if (config_.dimension != 3) {
        throw std::runtime_error("init_3d_glider() requires 3D");
    }

    grid_3d_->fill(0.0f);
    const int N = config_.grid_size;
    const float center = static_cast<float>(N) / 2.0f;
    const float kernel_R = static_cast<float>(config_.kernel.radius);

    // Two compact seeds offset along x, different intensities → asymmetry
    place_compact_seed(*grid_3d_, N, center, center, center - kernel_R * 0.1f,
                       kernel_R, 0.9f);
    place_compact_seed(*grid_3d_, N, center, center, center + kernel_R * 0.15f,
                       kernel_R, 0.4f);

    iteration_ = 0;
}

// ---------------------------------------------------------------------------
// init_3d_multi() -- Multiple compact seeds in tetrahedral arrangement
//
// Places 4 compact seeds separated by 2.5*R.  At this distance, the seeds
// are within kernel interaction range and their potentials overlap, creating
// complex interaction dynamics: merging, orbiting, or scattering.
// ---------------------------------------------------------------------------
void Lenia::init_3d_multi() {
    if (config_.dimension != 3) {
        throw std::runtime_error("init_3d_multi() requires 3D");
    }

    grid_3d_->fill(0.0f);
    const int N = config_.grid_size;
    const float c = static_cast<float>(N) / 2.0f;
    const float kernel_R = static_cast<float>(config_.kernel.radius);
    const float sep = kernel_R * 2.5f;

    // Tetrahedral arrangement around center
    place_compact_seed(*grid_3d_, N, c + sep,   c,              c,
                       kernel_R, 0.9f);
    place_compact_seed(*grid_3d_, N, c - sep/2, c + sep*0.87f,  c,
                       kernel_R, 0.8f);
    place_compact_seed(*grid_3d_, N, c - sep/2, c - sep*0.87f,  c,
                       kernel_R, 0.85f);
    place_compact_seed(*grid_3d_, N, c,         c,              c + sep*0.8f,
                       kernel_R, 0.7f);

    iteration_ = 0;
}

// ---------------------------------------------------------------------------
// init_3d_shell() -- Compact spherical seed
//
// Uses shell_r = 0.2*R, thickness = 0.2*R — a dense ball that produces the
// right convolution value for the growth function.  Under appropriate
// parameters, the seed can evolve into pulsing, deforming, or collapsing
// structures.
// ---------------------------------------------------------------------------
void Lenia::init_3d_shell() {
    if (config_.dimension != 3) {
        throw std::runtime_error("init_3d_shell() requires 3D");
    }

    grid_3d_->fill(0.0f);
    const int N = config_.grid_size;
    const float center = static_cast<float>(N) / 2.0f;
    const float kernel_R = static_cast<float>(config_.kernel.radius);
    const float shell_r = kernel_R * 0.2f;
    const float thickness = kernel_R * 0.2f;
    const float inv_2t2 = 1.0f / (2.0f * thickness * thickness);

    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float dz = static_cast<float>(k) - center;
                float dy = static_cast<float>(i) - center;
                float dx = static_cast<float>(j) - center;
                float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

                float diff = dist - shell_r;
                float val = 0.85f * std::exp(-diff * diff * inv_2t2);
                if (val > 0.01f) {
                    grid_3d_->at(k, i, j) = val;
                }
            }
        }
    }
    iteration_ = 0;
}

// ---------------------------------------------------------------------------
// init_3d_dipole() -- Two compact seeds of different intensity
//
// Asymmetric pair separated by 0.5*R: a faint seed next to a bright one.
// The imbalance can cause the pair to rotate, drift, or oscillate
// depending on the growth parameters, similar to biological cell division.
// ---------------------------------------------------------------------------
void Lenia::init_3d_dipole() {
    if (config_.dimension != 3) {
        throw std::runtime_error("init_3d_dipole() requires 3D");
    }

    grid_3d_->fill(0.0f);
    const int N = config_.grid_size;
    const float c = static_cast<float>(N) / 2.0f;
    const float kernel_R = static_cast<float>(config_.kernel.radius);
    const float gap = kernel_R * 0.5f;

    // Faint seed
    place_compact_seed(*grid_3d_, N, c - gap, c, c,
                       kernel_R, 0.5f);
    // Bright seed
    place_compact_seed(*grid_3d_, N, c + gap, c, c,
                       kernel_R, 0.95f);

    iteration_ = 0;
}

// ---------------------------------------------------------------------------
// set_growth_params()
// ---------------------------------------------------------------------------
void Lenia::set_growth_params(const GrowthParams& params) {
    config_.growth = params;
}

// ---------------------------------------------------------------------------
// set_kernel_params() -- triggers kernel recomputation
// ---------------------------------------------------------------------------
void Lenia::set_kernel_params(const KernelParams& params) {
    config_.kernel = params;
    rebuild_kernel();
}

// ---------------------------------------------------------------------------
// set_time_step()
// ---------------------------------------------------------------------------
void Lenia::set_time_step(int T) {
    assert(T > 0);
    config_.T = T;
    dt_ = 1.0f / static_cast<float>(T);
}

// ---------------------------------------------------------------------------
// rebuild_kernel() -- regenerate kernel and recompute FFT spectrum
// ---------------------------------------------------------------------------
void Lenia::rebuild_kernel() {
    int N = config_.grid_size;

    if (config_.dimension == 2) {
        Grid<2> kern = generate_kernel_2d(N, N, config_.kernel);
        kernel_spectrum_2d_ = precompute_kernel_spectrum(kern, config_.num_threads);

        extra_spectra_2d_.clear();
        for (const auto& kgp : config_.extra_kernels) {
            Grid<2> ek = generate_kernel_2d(N, N, kgp.kernel);
            extra_spectra_2d_.push_back(
                precompute_kernel_spectrum(ek, config_.num_threads));
        }
    } else {
        Grid<3> kern = generate_kernel_3d(N, N, N, config_.kernel);
        kernel_spectrum_3d_ = precompute_kernel_spectrum(kern, config_.num_threads);

        extra_spectra_3d_.clear();
        for (const auto& kgp : config_.extra_kernels) {
            Grid<3> ek = generate_kernel_3d(N, N, N, kgp.kernel);
            extra_spectra_3d_.push_back(
                precompute_kernel_spectrum(ek, config_.num_threads));
        }
    }
}
