#include "fft_engine.h"

#include <cstring>
#include <cassert>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// One-time FFTW thread initialization (single-precision)
// ---------------------------------------------------------------------------
static std::once_flag fftw_threads_init_flag;
static std::mutex fftw_plan_mutex;

static void init_fftw_threads() {
    fftwf_init_threads();
}

// ---------------------------------------------------------------------------
// Constructor
// Allocates real and complex buffers, creates forward and inverse plans.
// Uses single-precision (fftwf) throughout — no float↔double conversion.
// ---------------------------------------------------------------------------
FFTEngine::FFTEngine(int dim, const int* shape, int num_threads)
    : dim_(dim),
      shape_(shape, shape + dim),
      real_buffer_(nullptr),
      spectrum_(nullptr),
      forward_plan_(nullptr),
      inverse_plan_(nullptr) {

    assert(dim == 2 || dim == 3);

    // Compute sizes
    real_size_ = 1;
    for (int d = 0; d < dim; ++d) {
        real_size_ *= static_cast<std::size_t>(shape_[d]);
    }

    // Spectrum size: product of all dims except last, times (last/2 + 1)
    spectrum_size_ = 1;
    for (int d = 0; d < dim - 1; ++d) {
        spectrum_size_ *= static_cast<std::size_t>(shape_[d]);
    }
    spectrum_size_ *= static_cast<std::size_t>(shape_[dim - 1] / 2 + 1);

    // Ensure FFTW threads are initialized exactly once
    std::call_once(fftw_threads_init_flag, init_fftw_threads);

    // Allocate aligned buffers (single precision)
    real_buffer_ = fftwf_alloc_real(real_size_);
    spectrum_    = fftwf_alloc_complex(spectrum_size_);

    assert(real_buffer_ != nullptr);
    assert(spectrum_ != nullptr);

    // Zero-initialize buffers
    std::memset(real_buffer_, 0, real_size_ * sizeof(float));
    std::memset(spectrum_, 0, spectrum_size_ * sizeof(fftwf_complex));

    // Plan creation is not thread-safe in FFTW, so we lock
    {
        std::lock_guard<std::mutex> lock(fftw_plan_mutex);
        fftwf_plan_with_nthreads(num_threads);

        forward_plan_ = fftwf_plan_dft_r2c(
            dim, shape_.data(), real_buffer_, spectrum_, FFTW_MEASURE);

        inverse_plan_ = fftwf_plan_dft_c2r(
            dim, shape_.data(), spectrum_, real_buffer_, FFTW_MEASURE);
    }

    assert(forward_plan_ != nullptr);
    assert(inverse_plan_ != nullptr);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
FFTEngine::~FFTEngine() {
    if (forward_plan_) {
        std::lock_guard<std::mutex> lock(fftw_plan_mutex);
        fftwf_destroy_plan(forward_plan_);
    }
    if (inverse_plan_) {
        std::lock_guard<std::mutex> lock(fftw_plan_mutex);
        fftwf_destroy_plan(inverse_plan_);
    }
    if (real_buffer_) {
        fftwf_free(real_buffer_);
    }
    if (spectrum_) {
        fftwf_free(spectrum_);
    }
}

// ---------------------------------------------------------------------------
// forward() -- copy float input directly to float buffer, execute forward FFT
// No float↔double conversion needed — buffer is already float.
// ---------------------------------------------------------------------------
void FFTEngine::forward(const float* input) {
    std::memcpy(real_buffer_, input, real_size_ * sizeof(float));
    fftwf_execute(forward_plan_);
}

// ---------------------------------------------------------------------------
// inverse() -- execute inverse FFT, normalize, copy to float output
// ---------------------------------------------------------------------------
void FFTEngine::inverse(float* output) {
    fftwf_execute(inverse_plan_);

    // FFTW computes unnormalized inverse, so divide by N
    float inv_n = 1.0f / static_cast<float>(real_size_);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < real_size_; ++i) {
        output[i] = real_buffer_[i] * inv_n;
    }
}

// ---------------------------------------------------------------------------
// save_spectrum() -- copy internal spectrum to external storage
// ---------------------------------------------------------------------------
void FFTEngine::save_spectrum(std::vector<std::complex<float>>& dest) const {
    dest.resize(spectrum_size_);
    for (std::size_t i = 0; i < spectrum_size_; ++i) {
        dest[i] = std::complex<float>(spectrum_[i][0], spectrum_[i][1]);
    }
}

// ---------------------------------------------------------------------------
// load_spectrum() -- overwrite internal spectrum from external storage
// ---------------------------------------------------------------------------
void FFTEngine::load_spectrum(const std::vector<std::complex<float>>& src) {
    assert(src.size() == spectrum_size_);
    for (std::size_t i = 0; i < spectrum_size_; ++i) {
        spectrum_[i][0] = src[i].real();
        spectrum_[i][1] = src[i].imag();
    }
}

// ---------------------------------------------------------------------------
// pointwise_multiply() -- element-wise complex multiplication (float)
// ---------------------------------------------------------------------------
void FFTEngine::pointwise_multiply(
    const std::vector<std::complex<float>>& kernel_spectrum) {

    assert(kernel_spectrum.size() == spectrum_size_);

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < spectrum_size_; ++i) {
        float a = spectrum_[i][0];  // real part
        float b = spectrum_[i][1];  // imaginary part
        float c = kernel_spectrum[i].real();
        float d = kernel_spectrum[i].imag();

        spectrum_[i][0] = a * c - b * d;
        spectrum_[i][1] = a * d + b * c;
    }
}
