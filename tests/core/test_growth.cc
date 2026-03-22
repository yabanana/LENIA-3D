#include <gtest/gtest.h>
#include "core/growth.h"

#include <cmath>
#include <vector>

// ===========================================================================
// Scalar growth function tests
// ===========================================================================

TEST(Growth, PeakAtMuReturnsOne) {
    // G(mu) = 2 * exp(0) - 1 = 2 - 1 = 1
    GrowthParams params;
    params.mu = 0.15f;
    params.sigma = 0.015f;

    float result = growth(params.mu, params);
    EXPECT_NEAR(result, 1.0f, 1e-6f)
        << "Growth function should peak at 1.0 when u == mu";
}

TEST(Growth, FarFromMuReturnsNearNegativeOne) {
    // When u is far from mu, exp(...) -> 0, so G(u) -> -1
    GrowthParams params;
    params.mu = 0.15f;
    params.sigma = 0.015f;

    float result_zero = growth(0.0f, params);
    EXPECT_NEAR(result_zero, -1.0f, 0.01f)
        << "Growth function should be near -1 when u is far from mu";

    float result_one = growth(1.0f, params);
    EXPECT_NEAR(result_one, -1.0f, 0.01f)
        << "Growth function should be near -1 when u is far from mu";
}

TEST(Growth, SymmetryAroundMu) {
    // G(mu + delta) == G(mu - delta)
    GrowthParams params;
    params.mu = 0.15f;
    params.sigma = 0.015f;

    float delta = 0.005f;
    float g_plus  = growth(params.mu + delta, params);
    float g_minus = growth(params.mu - delta, params);

    EXPECT_NEAR(g_plus, g_minus, 1e-6f)
        << "Growth function should be symmetric around mu";
}

TEST(Growth, MonotonicallyDecreasingFromPeak) {
    // G should decrease as we move away from mu in either direction
    GrowthParams params;
    params.mu = 0.15f;
    params.sigma = 0.015f;

    float g_at_mu = growth(params.mu, params);
    float g_half_sigma = growth(params.mu + params.sigma * 0.5f, params);
    float g_one_sigma  = growth(params.mu + params.sigma, params);
    float g_two_sigma  = growth(params.mu + params.sigma * 2.0f, params);

    EXPECT_GT(g_at_mu, g_half_sigma);
    EXPECT_GT(g_half_sigma, g_one_sigma);
    EXPECT_GT(g_one_sigma, g_two_sigma);
}

TEST(Growth, RangeIsBetweenMinusOneAndOne) {
    // G(u) should always be in [-1, 1]
    GrowthParams params;
    params.mu = 0.15f;
    params.sigma = 0.015f;

    for (float u = 0.0f; u <= 1.0f; u += 0.01f) {
        float g = growth(u, params);
        EXPECT_GE(g, -1.0f) << "Growth should be >= -1 at u=" << u;
        EXPECT_LE(g, 1.0f)  << "Growth should be <= 1 at u=" << u;
    }
}

TEST(Growth, DifferentParameters) {
    // Test with different mu/sigma values
    GrowthParams params;
    params.mu = 0.5f;
    params.sigma = 0.1f;

    float g_peak = growth(0.5f, params);
    EXPECT_NEAR(g_peak, 1.0f, 1e-6f);

    float g_far = growth(0.0f, params);
    EXPECT_LT(g_far, 0.0f)
        << "Growth at 0 should be negative when mu=0.5, sigma=0.1";
}

// ===========================================================================
// Batch growth function tests
// ===========================================================================

TEST(GrowthBatch, MatchesScalarVersion) {
    GrowthParams params;
    params.mu = 0.15f;
    params.sigma = 0.015f;

    const std::size_t n = 100;
    std::vector<float> input(n);
    std::vector<float> output_batch(n);

    for (std::size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i) / static_cast<float>(n);
    }

    growth_batch(input.data(), output_batch.data(), n, params);

    for (std::size_t i = 0; i < n; ++i) {
        float expected = growth(input[i], params);
        EXPECT_NEAR(output_batch[i], expected, 1e-6f)
            << "Batch output differs from scalar at index " << i;
    }
}

TEST(GrowthBatch, HandlesEmptyInput) {
    GrowthParams params;
    growth_batch(nullptr, nullptr, 0, params);
    // Should not crash with zero-length input
}

TEST(GrowthBatch, InPlaceOperation) {
    // growth_batch should work when input and output are the same buffer
    GrowthParams params;
    params.mu = 0.15f;
    params.sigma = 0.015f;

    const std::size_t n = 50;
    std::vector<float> data(n);
    std::vector<float> expected(n);

    for (std::size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i) / static_cast<float>(n);
        expected[i] = growth(data[i], params);
    }

    // Apply in-place
    growth_batch(data.data(), data.data(), n, params);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(data[i], expected[i], 1e-6f)
            << "In-place batch failed at index " << i;
    }
}

TEST(GrowthBatch, LargeArray) {
    // Test with a larger array to exercise potential OpenMP path
    GrowthParams params;
    params.mu = 0.15f;
    params.sigma = 0.015f;

    const std::size_t n = 10000;
    std::vector<float> input(n);
    std::vector<float> output(n);

    for (std::size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(i) / static_cast<float>(n);
    }

    growth_batch(input.data(), output.data(), n, params);

    // Spot-check a few values
    EXPECT_NEAR(output[0], growth(input[0], params), 1e-5f);
    EXPECT_NEAR(output[n / 2], growth(input[n / 2], params), 1e-5f);
    EXPECT_NEAR(output[n - 1], growth(input[n - 1], params), 1e-5f);
}
