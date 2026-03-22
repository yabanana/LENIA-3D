#include <gtest/gtest.h>
#include "core/fft_engine.h"

#include <cmath>
#include <complex>
#include <vector>

// ===========================================================================
// 2D FFT Tests
// ===========================================================================

TEST(FFT2D, ForwardInverseRecoverConstantField) {
    // A constant field should survive a forward+inverse round-trip.
    int N = 16;
    int shape[2] = {N, N};
    FFTEngine engine(2, shape, 1);

    std::size_t total = static_cast<std::size_t>(N) * N;
    std::vector<float> input(total, 0.5f);
    std::vector<float> output(total, 0.0f);

    engine.forward(input.data());

    // For a constant field, only the DC component should be nonzero.
    // DC component (index 0) should be N*N * 0.5 = 128
    float dc_real = engine.spectrum()[0][0];
    float dc_imag = engine.spectrum()[0][1];
    EXPECT_NEAR(dc_real, total * 0.5, 1e-3);
    EXPECT_NEAR(dc_imag, 0.0, 1e-5);

    engine.inverse(output.data());

    for (std::size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(output[i], 0.5f, 1e-4f)
            << "Constant field not recovered at index " << i;
    }
}

TEST(FFT2D, ForwardInverseRecoverImpulse) {
    // A single impulse (delta function) at (0,0) should survive round-trip.
    int N = 8;
    int shape[2] = {N, N};
    FFTEngine engine(2, shape, 1);

    std::size_t total = static_cast<std::size_t>(N) * N;
    std::vector<float> input(total, 0.0f);
    input[0] = 1.0f;  // Delta at origin

    std::vector<float> output(total, 0.0f);

    engine.forward(input.data());
    engine.inverse(output.data());

    for (std::size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(output[i], input[i], 1e-4f)
            << "Impulse not recovered at index " << i;
    }
}

TEST(FFT2D, ForwardInverseRecoverArbitrarySignal) {
    // An arbitrary signal should survive a round-trip within floating point tolerance.
    int N = 16;
    int shape[2] = {N, N};
    FFTEngine engine(2, shape, 1);

    std::size_t total = static_cast<std::size_t>(N) * N;
    std::vector<float> input(total);

    // Fill with a known pattern: sine wave
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float x = static_cast<float>(i) / N;
            float y = static_cast<float>(j) / N;
            input[i * N + j] = std::sin(2.0f * 3.14159265f * x)
                              + 0.5f * std::cos(4.0f * 3.14159265f * y);
        }
    }

    std::vector<float> output(total, 0.0f);

    engine.forward(input.data());
    engine.inverse(output.data());

    for (std::size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(output[i], input[i], 1e-4f)
            << "Arbitrary signal not recovered at index " << i;
    }
}

TEST(FFT2D, PointwiseMultiplyWithIdentity) {
    // Multiplying the spectrum by 1+0i should leave the signal unchanged.
    int N = 8;
    int shape[2] = {N, N};
    FFTEngine engine(2, shape, 1);

    std::size_t total = static_cast<std::size_t>(N) * N;
    std::vector<float> input(total);

    // Some input signal
    for (std::size_t i = 0; i < total; ++i) {
        input[i] = static_cast<float>(i) / static_cast<float>(total);
    }

    engine.forward(input.data());

    // Identity spectrum: all ones
    std::vector<std::complex<float>> identity(engine.spectrum_size(),
                                               std::complex<float>(1.0f, 0.0f));
    engine.pointwise_multiply(identity);

    std::vector<float> output(total, 0.0f);
    engine.inverse(output.data());

    for (std::size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(output[i], input[i], 1e-4f)
            << "Identity multiply changed signal at index " << i;
    }
}

TEST(FFT2D, PointwiseMultiplyWithZero) {
    // Multiplying the spectrum by 0 should produce an all-zero output.
    int N = 8;
    int shape[2] = {N, N};
    FFTEngine engine(2, shape, 1);

    std::size_t total = static_cast<std::size_t>(N) * N;
    std::vector<float> input(total, 0.7f);

    engine.forward(input.data());

    std::vector<std::complex<float>> zero_spectrum(engine.spectrum_size(),
                                                    std::complex<float>(0.0f, 0.0f));
    engine.pointwise_multiply(zero_spectrum);

    std::vector<float> output(total, 999.0f);
    engine.inverse(output.data());

    for (std::size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(output[i], 0.0f, 1e-6f);
    }
}

TEST(FFT2D, SpectrumSizeIsCorrect) {
    int N = 32;
    int shape[2] = {N, N};
    FFTEngine engine(2, shape, 1);

    // For real-to-complex: N * (N/2 + 1)
    std::size_t expected = static_cast<std::size_t>(N) * (N / 2 + 1);
    EXPECT_EQ(engine.spectrum_size(), expected);
    EXPECT_EQ(engine.real_size(), static_cast<std::size_t>(N) * N);
}

// ===========================================================================
// 3D FFT Tests
// ===========================================================================

TEST(FFT3D, ForwardInverseRecoverConstantField) {
    int N = 8;
    int shape[3] = {N, N, N};
    FFTEngine engine(3, shape, 1);

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<float> input(total, 0.3f);
    std::vector<float> output(total, 0.0f);

    engine.forward(input.data());
    engine.inverse(output.data());

    for (std::size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(output[i], 0.3f, 1e-4f)
            << "3D constant field not recovered at index " << i;
    }
}

TEST(FFT3D, ForwardInverseRecoverImpulse) {
    int N = 8;
    int shape[3] = {N, N, N};
    FFTEngine engine(3, shape, 1);

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<float> input(total, 0.0f);
    input[0] = 1.0f;

    std::vector<float> output(total, 0.0f);

    engine.forward(input.data());
    engine.inverse(output.data());

    for (std::size_t i = 0; i < total; ++i) {
        EXPECT_NEAR(output[i], input[i], 1e-4f)
            << "3D impulse not recovered at index " << i;
    }
}

TEST(FFT3D, SpectrumSizeIsCorrect) {
    int N = 16;
    int shape[3] = {N, N, N};
    FFTEngine engine(3, shape, 1);

    // For real-to-complex 3D: N * N * (N/2 + 1)
    std::size_t expected = static_cast<std::size_t>(N) * N * (N / 2 + 1);
    EXPECT_EQ(engine.spectrum_size(), expected);
    EXPECT_EQ(engine.real_size(), static_cast<std::size_t>(N) * N * N);
}
