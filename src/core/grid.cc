#include "grid.h"

#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
template<int Dim>
Grid<Dim>::Grid(const std::array<int, Dim>& shape) : shape_(shape) {
    std::size_t total = 1;
    for (int d = 0; d < Dim; ++d) {
        assert(shape[d] > 0);
        total *= static_cast<std::size_t>(shape[d]);
    }
    data_.resize(total, 0.0f);
}

// ---------------------------------------------------------------------------
// Index linearization
// ---------------------------------------------------------------------------
template<int Dim>
std::size_t Grid<Dim>::linearize(const std::array<int, Dim>& idx) const {
    // Row-major order: last index varies fastest
    std::size_t linear = 0;
    std::size_t stride = 1;
    for (int d = Dim - 1; d >= 0; --d) {
        assert(idx[d] >= 0 && idx[d] < shape_[d]);
        linear += static_cast<std::size_t>(idx[d]) * stride;
        stride *= static_cast<std::size_t>(shape_[d]);
    }
    return linear;
}

// ---------------------------------------------------------------------------
// operator()
// ---------------------------------------------------------------------------
template<int Dim>
float& Grid<Dim>::operator()(const std::array<int, Dim>& idx) {
    return data_[linearize(idx)];
}

template<int Dim>
const float& Grid<Dim>::operator()(const std::array<int, Dim>& idx) const {
    return data_[linearize(idx)];
}

// ---------------------------------------------------------------------------
// at(i, j) -- 2D convenience
// Note: we use a runtime assert instead of static_assert because explicit
// template instantiation instantiates all methods for all dimensions.
// The 2D at(i,j) body is still valid C++ even when Dim!=2; the assert
// simply prevents it from being called at runtime with the wrong dimension.
// ---------------------------------------------------------------------------
template<int Dim>
float& Grid<Dim>::at(int i, int j) {
    assert(Dim == 2 && "at(i,j) is only valid for 2D grids");
    return data_[static_cast<std::size_t>(i) * static_cast<std::size_t>(shape_[1])
                 + static_cast<std::size_t>(j)];
}

template<int Dim>
const float& Grid<Dim>::at(int i, int j) const {
    assert(Dim == 2 && "at(i,j) is only valid for 2D grids");
    return data_[static_cast<std::size_t>(i) * static_cast<std::size_t>(shape_[1])
                 + static_cast<std::size_t>(j)];
}

// ---------------------------------------------------------------------------
// at(i, j, k) -- 3D convenience
// ---------------------------------------------------------------------------
template<int Dim>
float& Grid<Dim>::at(int i, int j, int k) {
    assert(Dim == 3 && "at(i,j,k) is only valid for 3D grids");
    // When Dim < 3 the shape_[2] access is technically out-of-range for the
    // std::array, but this code path is unreachable at runtime due to the
    // assert above. We guard with a constexpr check to avoid UB.
    if constexpr (Dim >= 3) {
        return data_[static_cast<std::size_t>(i)
                         * static_cast<std::size_t>(shape_[1])
                         * static_cast<std::size_t>(shape_[2])
                     + static_cast<std::size_t>(j)
                         * static_cast<std::size_t>(shape_[2])
                     + static_cast<std::size_t>(k)];
    } else {
        // Unreachable: the assert above fires before we get here.
        return data_[0];
    }
}

template<int Dim>
const float& Grid<Dim>::at(int i, int j, int k) const {
    assert(Dim == 3 && "at(i,j,k) is only valid for 3D grids");
    if constexpr (Dim >= 3) {
        return data_[static_cast<std::size_t>(i)
                         * static_cast<std::size_t>(shape_[1])
                         * static_cast<std::size_t>(shape_[2])
                     + static_cast<std::size_t>(j)
                         * static_cast<std::size_t>(shape_[2])
                     + static_cast<std::size_t>(k)];
    } else {
        return data_[0];
    }
}

// ---------------------------------------------------------------------------
// total_size
// ---------------------------------------------------------------------------
template<int Dim>
std::size_t Grid<Dim>::total_size() const {
    std::size_t total = 1;
    for (int d = 0; d < Dim; ++d) {
        total *= static_cast<std::size_t>(shape_[d]);
    }
    return total;
}

// ---------------------------------------------------------------------------
// fill
// ---------------------------------------------------------------------------
template<int Dim>
void Grid<Dim>::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

// ---------------------------------------------------------------------------
// fill_random
// ---------------------------------------------------------------------------
template<int Dim>
void Grid<Dim>::fill_random(float low, float high, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    for (auto& v : data_) {
        v = dist(rng);
    }
}

// ---------------------------------------------------------------------------
// place_blob -- circular (2D) or spherical (3D) blob
// ---------------------------------------------------------------------------
template<int Dim>
void Grid<Dim>::place_blob(const std::array<int, Dim>& center,
                           int radius, float value) {
    int r2 = radius * radius;

    if constexpr (Dim == 2) {
        for (int di = -radius; di <= radius; ++di) {
            for (int dj = -radius; dj <= radius; ++dj) {
                if (di * di + dj * dj <= r2) {
                    int i = (center[0] + di + shape_[0]) % shape_[0];
                    int j = (center[1] + dj + shape_[1]) % shape_[1];
                    at(i, j) = value;
                }
            }
        }
    } else if constexpr (Dim == 3) {
        for (int di = -radius; di <= radius; ++di) {
            for (int dj = -radius; dj <= radius; ++dj) {
                for (int dk = -radius; dk <= radius; ++dk) {
                    if (di * di + dj * dj + dk * dk <= r2) {
                        int i = (center[0] + di + shape_[0]) % shape_[0];
                        int j = (center[1] + dj + shape_[1]) % shape_[1];
                        int k = (center[2] + dk + shape_[2]) % shape_[2];
                        at(i, j, k) = value;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// total_mass
// ---------------------------------------------------------------------------
template<int Dim>
double Grid<Dim>::total_mass() const {
    double sum = 0.0;
    for (const auto& v : data_) {
        sum += static_cast<double>(v);
    }
    return sum;
}

// ---------------------------------------------------------------------------
// clip
// ---------------------------------------------------------------------------
template<int Dim>
void Grid<Dim>::clip() {
    for (auto& v : data_) {
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
template class Grid<2>;
template class Grid<3>;
