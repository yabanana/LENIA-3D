#pragma once

#include "grid.h"

#include <vector>
#include <complex>

// ---------------------------------------------------------------------------
// Kernel core function type (Chan's "kn" parameter).
//   kn=0: Polynomial bump  (4*r*(1-r))^alpha  — used by classic Orbium
//   kn=1: Exponential bump  exp(4 - 1/(r*(1-r)))  — used by 312 patterns
// ---------------------------------------------------------------------------
enum class KernelCoreFunc : int {
    Polynomial  = 0,   // kn=0: (4*r*(1-r))^alpha
    Exponential = 1    // kn=1: exp(4 - 1/(r*(1-r)))
};

// ---------------------------------------------------------------------------
// Kernel parameters following Chan 2019 "Lenia - Biology of Artificial Life".
//
// The kernel K(x) is an annular function that depends on the distance |x|
// from the cell.  For a normalized distance r = |x| / R:
//
//   K_shell(r) = (r < 1) * kernel_core(B*r mod 1) * beta[floor(B*r)]
//
// where B = number of rings, beta = weights per ring, and
//   kernel_core depends on core_func:
//     Polynomial:  (4*r*(1-r))^alpha  (smooth polynomial bump)
//     Exponential: exp(4 - 1/(r*(1-r)))  (smooth exponential bump)
//
// For the canonical Orbium: B=1, beta=[1], alpha=4, core_func=Polynomial, R=13.
// For Chan 312 patterns: core_func=Exponential, various R and beta.
// ---------------------------------------------------------------------------
struct KernelParams {
    int   radius      = 13;              // Kernel radius R
    std::vector<float> beta = {1.0f};    // Weight per ring (size = num_rings)
    int   core_alpha  = 4;               // Exponent for polynomial core
    KernelCoreFunc core_func = KernelCoreFunc::Polynomial;  // kn type

    int num_rings() const { return static_cast<int>(beta.size()); }
};

// Generate 2D annular kernel in spatial domain, centered for FFT (DC at corner)
Grid<2> generate_kernel_2d(int rows, int cols, const KernelParams& params);

// Generate 3D annular kernel in spatial domain, centered for FFT (DC at corner)
Grid<3> generate_kernel_3d(int d, int r, int c, const KernelParams& params);

// Precompute kernel FFT spectrum (2D)
std::vector<std::complex<float>> precompute_kernel_spectrum(
    const Grid<2>& kernel, int num_threads = 1);

// Precompute kernel FFT spectrum (3D)
std::vector<std::complex<float>> precompute_kernel_spectrum(
    const Grid<3>& kernel, int num_threads = 1);
