#include "kernel.h"
#include "fft_engine.h"

#include <cmath>
#include <cassert>
#include <algorithm>

// ---------------------------------------------------------------------------
// Kernel core function from Chan 2019.
//
// kn=0 (Polynomial):  (4*r*(1-r))^alpha   — smooth bump, peaks at r=0.5
// kn=1 (Exponential): exp(4 - 1/(r*(1-r))) — smooth C∞ bump, peaks at r=0.5
//
// Both return 0 at r=0 and r=1, and 1.0 at r=0.5.
// The exponential version is narrower in the tails.
// ---------------------------------------------------------------------------
static float kernel_core(float r, int alpha, KernelCoreFunc func) {
    if (r <= 0.0f || r >= 1.0f) {
        return 0.0f;
    }
    if (func == KernelCoreFunc::Exponential) {
        float denom = r * (1.0f - r);
        return std::exp(4.0f - 1.0f / denom);
    }
    // Polynomial
    float base = 4.0f * r * (1.0f - r);
    float result = 1.0f;
    for (int i = 0; i < alpha; ++i) {
        result *= base;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Kernel shell function from Chan 2019.
//
// For B rings with weights beta[0..B-1]:
//   K_shell(rn) = kernel_core( (B * rn) mod 1 ) * beta[floor(B * rn)]
//
// This divides the radial range [0,1) into B equal segments.  Each segment
// gets a copy of kernel_core scaled by its beta weight.
//
// For the canonical Orbium (B=1, beta=[1]):
//   K_shell(rn) = kernel_core(rn) * 1.0 = (4*rn*(1-rn))^4
// ---------------------------------------------------------------------------
static float kernel_shell(float normalized_r, const std::vector<float>& beta,
                          int alpha, KernelCoreFunc func) {
    if (normalized_r >= 1.0f || normalized_r <= 0.0f) {
        return 0.0f;
    }

    int B = static_cast<int>(beta.size());
    float Br = static_cast<float>(B) * normalized_r;
    int ring = std::min(static_cast<int>(Br), B - 1);
    float local_r = std::fmod(Br, 1.0f);

    // Clamp local_r to [0, 1] (handles floating-point edge cases)
    local_r = std::min(local_r, 1.0f);

    return kernel_core(local_r, alpha, func) * beta[ring];
}

// ---------------------------------------------------------------------------
// generate_kernel_2d
// Creates a 2D annular kernel ready for FFT convolution.
// The kernel is placed with its center at (0,0) using wraparound indexing,
// so that FFTW can perform circular convolution directly.
// ---------------------------------------------------------------------------
Grid<2> generate_kernel_2d(int rows, int cols, const KernelParams& params) {
    Grid<2> kernel(std::array<int, 2>{rows, cols});
    kernel.fill(0.0f);

    int R = params.radius;
    double sum = 0.0;

    for (int di = -R; di <= R; ++di) {
        for (int dj = -R; dj <= R; ++dj) {
            float dist = std::sqrt(static_cast<float>(di * di + dj * dj));
            if (dist > static_cast<float>(R)) continue;

            float norm_r = dist / static_cast<float>(R);
            float val = kernel_shell(norm_r, params.beta, params.core_alpha, params.core_func);

            if (val < 1e-10f) continue;

            // Wrap to FFT-friendly layout: center at (0,0) with wraparound
            int i = ((di % rows) + rows) % rows;
            int j = ((dj % cols) + cols) % cols;
            kernel.at(i, j) += val;
            sum += static_cast<double>(val);
        }
    }

    // Normalize so the kernel sums to 1
    if (sum > 1e-15) {
        float inv_sum = static_cast<float>(1.0 / sum);
        float* d = kernel.data();
        std::size_t n = kernel.total_size();
        for (std::size_t i = 0; i < n; ++i) {
            d[i] *= inv_sum;
        }
    }

    return kernel;
}

// ---------------------------------------------------------------------------
// generate_kernel_3d
// Creates a 3D annular (spherical shell) kernel ready for FFT convolution.
// ---------------------------------------------------------------------------
Grid<3> generate_kernel_3d(int depth, int rows, int cols,
                           const KernelParams& params) {
    Grid<3> kernel(std::array<int, 3>{depth, rows, cols});
    kernel.fill(0.0f);

    int R = params.radius;
    double sum = 0.0;

    for (int dd = -R; dd <= R; ++dd) {
        for (int di = -R; di <= R; ++di) {
            for (int dj = -R; dj <= R; ++dj) {
                float dist = std::sqrt(static_cast<float>(
                    dd * dd + di * di + dj * dj));
                if (dist > static_cast<float>(R)) continue;

                float norm_r = dist / static_cast<float>(R);
                float val = kernel_shell(norm_r, params.beta, params.core_alpha, params.core_func);

                if (val < 1e-10f) continue;

                int id = ((dd % depth) + depth) % depth;
                int ir = ((di % rows) + rows) % rows;
                int ic = ((dj % cols) + cols) % cols;
                kernel.at(id, ir, ic) += val;
                sum += static_cast<double>(val);
            }
        }
    }

    // Normalize so the kernel sums to 1
    if (sum > 1e-15) {
        float inv_sum = static_cast<float>(1.0 / sum);
        float* d = kernel.data();
        std::size_t n = kernel.total_size();
        for (std::size_t i = 0; i < n; ++i) {
            d[i] *= inv_sum;
        }
    }

    return kernel;
}

// ---------------------------------------------------------------------------
// precompute_kernel_spectrum (2D)
// Computes the FFT of the kernel and returns the complex spectrum as a
// std::vector for later pointwise multiplication.
// ---------------------------------------------------------------------------
std::vector<std::complex<float>> precompute_kernel_spectrum(
    const Grid<2>& kernel, int num_threads) {

    const auto& s = kernel.shape();
    int shape[2] = {s[0], s[1]};

    FFTEngine engine(2, shape, num_threads);
    engine.forward(kernel.data());

    std::size_t n = engine.spectrum_size();
    std::vector<std::complex<float>> result(n);
    const fftwf_complex* spec = engine.spectrum();

    for (std::size_t i = 0; i < n; ++i) {
        result[i] = std::complex<float>(spec[i][0], spec[i][1]);
    }

    return result;
}

// ---------------------------------------------------------------------------
// precompute_kernel_spectrum (3D)
// ---------------------------------------------------------------------------
std::vector<std::complex<float>> precompute_kernel_spectrum(
    const Grid<3>& kernel, int num_threads) {

    const auto& s = kernel.shape();
    int shape[3] = {s[0], s[1], s[2]};

    FFTEngine engine(3, shape, num_threads);
    engine.forward(kernel.data());

    std::size_t n = engine.spectrum_size();
    std::vector<std::complex<float>> result(n);
    const fftwf_complex* spec = engine.spectrum();

    for (std::size_t i = 0; i < n; ++i) {
        result[i] = std::complex<float>(spec[i][0], spec[i][1]);
    }

    return result;
}
