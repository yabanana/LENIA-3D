#pragma once

#include <cstddef>

struct GrowthParams {
    float mu    = 0.15f;   // Center of growth function
    float sigma = 0.015f;  // Width of growth function
};

// G(u) = 2 * exp(-((u - mu)^2) / (2 * sigma^2)) - 1
// Returns value in [-1, 1]
float growth(float u, const GrowthParams& params);

// Vectorized version for OpenMP
void growth_batch(const float* input, float* output, std::size_t n,
                  const GrowthParams& params);
