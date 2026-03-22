#include "growth.h"

#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Scalar growth function
// G(u) = 2 * exp( -((u - mu)^2) / (2 * sigma^2) ) - 1
// Maps the neighborhood potential u to a growth/decay value in [-1, +1].
// When u is near mu, growth is near +1 (cell state increases).
// When u is far from mu, growth is near -1 (cell state decreases).
// ---------------------------------------------------------------------------
float growth(float u, const GrowthParams& params) {
    float diff = u - params.mu;
    float exponent = -(diff * diff) / (2.0f * params.sigma * params.sigma);
    return 2.0f * std::exp(exponent) - 1.0f;
}

// ---------------------------------------------------------------------------
// Batch growth function with OpenMP parallelization
// Applies the growth function element-wise over an array.
// ---------------------------------------------------------------------------
void growth_batch(const float* input, float* output, std::size_t n,
                  const GrowthParams& params) {
    float mu    = params.mu;
    float sigma = params.sigma;
    float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        float diff = input[i] - mu;
        float exponent = -(diff * diff) * inv_2sigma2;
        output[i] = 2.0f * std::exp(exponent) - 1.0f;
    }
}
