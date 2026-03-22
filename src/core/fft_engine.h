#pragma once

#include <fftw3.h>
#include <vector>
#include <complex>
#include <cstddef>

class FFTEngine {
public:
    // dim=2: shape={rows,cols}, dim=3: shape={depth,rows,cols}
    FFTEngine(int dim, const int* shape, int num_threads = 1);
    ~FFTEngine();

    FFTEngine(const FFTEngine&) = delete;
    FFTEngine& operator=(const FFTEngine&) = delete;

    // Forward FFT: real float input -> complex output
    void forward(const float* input);

    // Inverse FFT: complex input -> real float output (normalized)
    void inverse(float* output);

    // Pointwise multiply current spectrum with kernel spectrum
    void pointwise_multiply(const std::vector<std::complex<float>>& kernel_spectrum);

    // Access spectrum for external operations
    fftwf_complex* spectrum() { return spectrum_; }
    const fftwf_complex* spectrum() const { return spectrum_; }

    // Copy current spectrum to external storage (for multi-kernel support)
    void save_spectrum(std::vector<std::complex<float>>& dest) const;

    // Load spectrum from external storage (overwrite internal)
    void load_spectrum(const std::vector<std::complex<float>>& src);

    std::size_t spectrum_size() const { return spectrum_size_; }
    std::size_t real_size() const { return real_size_; }

private:
    int dim_;
    std::vector<int> shape_;
    std::size_t real_size_;
    std::size_t spectrum_size_;

    float* real_buffer_;
    fftwf_complex* spectrum_;
    fftwf_plan forward_plan_;
    fftwf_plan inverse_plan_;
};
