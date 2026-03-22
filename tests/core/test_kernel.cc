#include <gtest/gtest.h>
#include "core/kernel.h"

#include <cmath>
#include <numeric>

// ===========================================================================
// 2D Kernel Tests
// ===========================================================================

TEST(Kernel2D, KernelSumsToOne) {
    KernelParams params;
    params.radius = 13;

    Grid<2> kernel = generate_kernel_2d(64, 64, params);

    double sum = 0.0;
    for (std::size_t i = 0; i < kernel.total_size(); ++i) {
        sum += static_cast<double>(kernel.data()[i]);
    }

    EXPECT_NEAR(sum, 1.0, 1e-6)
        << "Normalized kernel should sum to approximately 1.0";
}

TEST(Kernel2D, KernelHasCorrectShape) {
    KernelParams params;
    params.radius = 10;

    Grid<2> kernel = generate_kernel_2d(64, 64, params);

    EXPECT_EQ(kernel.size(0), 64);
    EXPECT_EQ(kernel.size(1), 64);
}

TEST(Kernel2D, KernelIsNonNegative) {
    KernelParams params;
    params.radius = 13;

    Grid<2> kernel = generate_kernel_2d(64, 64, params);

    for (std::size_t i = 0; i < kernel.total_size(); ++i) {
        EXPECT_GE(kernel.data()[i], 0.0f)
            << "Kernel values should be non-negative at index " << i;
    }
}

TEST(Kernel2D, KernelHasAnnularStructure) {
    // The kernel should have its peak at around half of R from the center.
    KernelParams params;
    params.radius = 13;

    int N = 64;
    Grid<2> kernel = generate_kernel_2d(N, N, params);

    // The center value (at (0,0) in FFT layout) should be near zero
    // because the bell function returns 0 at normalized_r = 0
    float center_val = kernel.at(0, 0);
    EXPECT_LT(center_val, 0.01f)
        << "Center of annular kernel should be near zero";

    // The peak ring should be at approximately beta_center * R = 6.5 cells
    // from center. Check that values around distance 6-7 are higher than
    // values at distance 1-2.
    float val_near_center = kernel.at(1, 0);
    float val_at_ring = kernel.at(7, 0);  // Wrapped: row 7 in FFT layout

    // At ring distance the value should be appreciably larger than near center
    EXPECT_GT(val_at_ring, val_near_center)
        << "Values at ring distance should exceed values near center";
}

TEST(Kernel2D, KernelIsRadiallySymmetric) {
    KernelParams params;
    params.radius = 10;

    int N = 64;
    Grid<2> kernel = generate_kernel_2d(N, N, params);

    // Test symmetry: value at (d, 0) should equal value at (0, d)
    // because the kernel is radially symmetric.
    for (int d = 1; d <= params.radius; ++d) {
        int idx_row = d % N;
        int idx_col = d % N;

        float val_horizontal = kernel.at(0, idx_col);
        float val_vertical   = kernel.at(idx_row, 0);

        EXPECT_NEAR(val_horizontal, val_vertical, 1e-6f)
            << "Kernel should be radially symmetric at distance " << d;
    }

    // Also check diagonal symmetry: (d, 0) vs (-d, 0) i.e. (N-d, 0)
    for (int d = 1; d < params.radius && d < N; ++d) {
        float val_pos = kernel.at(d, 0);
        float val_neg = kernel.at(N - d, 0);

        EXPECT_NEAR(val_pos, val_neg, 1e-6f)
            << "Kernel should be symmetric across origin at distance " << d;
    }
}

TEST(Kernel2D, SpectrumHasCorrectSize) {
    KernelParams params;
    params.radius = 13;

    int N = 32;
    Grid<2> kernel = generate_kernel_2d(N, N, params);
    auto spectrum = precompute_kernel_spectrum(kernel, 1);

    // For a real-to-complex FFT of NxN, spectrum size = N * (N/2 + 1)
    std::size_t expected_size = static_cast<std::size_t>(N) * (N / 2 + 1);
    EXPECT_EQ(spectrum.size(), expected_size);
}

TEST(Kernel2D, SpectrumDCComponentMatchesSum) {
    KernelParams params;
    params.radius = 10;

    int N = 32;
    Grid<2> kernel = generate_kernel_2d(N, N, params);

    double spatial_sum = 0.0;
    for (std::size_t i = 0; i < kernel.total_size(); ++i) {
        spatial_sum += static_cast<double>(kernel.data()[i]);
    }

    auto spectrum = precompute_kernel_spectrum(kernel, 1);

    // DC component (index 0) should equal the sum of all spatial values
    double dc_real = spectrum[0].real();
    EXPECT_NEAR(dc_real, spatial_sum, 1e-4)
        << "DC component should equal the spatial sum of the kernel";

    // Imaginary part of DC should be ~0
    EXPECT_NEAR(spectrum[0].imag(), 0.0, 1e-10);
}

// ===========================================================================
// 3D Kernel Tests
// ===========================================================================

TEST(Kernel3D, KernelSumsToOne) {
    KernelParams params;
    params.radius = 5;

    Grid<3> kernel = generate_kernel_3d(16, 16, 16, params);

    double sum = 0.0;
    for (std::size_t i = 0; i < kernel.total_size(); ++i) {
        sum += static_cast<double>(kernel.data()[i]);
    }

    EXPECT_NEAR(sum, 1.0, 1e-5)
        << "Normalized 3D kernel should sum to approximately 1.0";
}

TEST(Kernel3D, KernelHasCorrectShape) {
    KernelParams params;
    params.radius = 5;

    Grid<3> kernel = generate_kernel_3d(16, 16, 16, params);

    EXPECT_EQ(kernel.size(0), 16);
    EXPECT_EQ(kernel.size(1), 16);
    EXPECT_EQ(kernel.size(2), 16);
}

TEST(Kernel3D, KernelIsNonNegative) {
    KernelParams params;
    params.radius = 5;

    Grid<3> kernel = generate_kernel_3d(16, 16, 16, params);

    for (std::size_t i = 0; i < kernel.total_size(); ++i) {
        EXPECT_GE(kernel.data()[i], 0.0f);
    }
}

TEST(Kernel3D, SpectrumHasCorrectSize) {
    KernelParams params;
    params.radius = 5;

    int N = 16;
    Grid<3> kernel = generate_kernel_3d(N, N, N, params);
    auto spectrum = precompute_kernel_spectrum(kernel, 1);

    // For a real-to-complex 3D FFT of NxNxN, size = N * N * (N/2 + 1)
    std::size_t expected_size = static_cast<std::size_t>(N) * N * (N / 2 + 1);
    EXPECT_EQ(spectrum.size(), expected_size);
}
