#pragma once

#include <vector>
#include <array>
#include <cstddef>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <random>
#include <functional>

template<int Dim>
class Grid {
public:
    Grid() = default;
    explicit Grid(const std::array<int, Dim>& shape);

    float& operator()(const std::array<int, Dim>& idx);
    const float& operator()(const std::array<int, Dim>& idx) const;

    // For 2D convenience
    float& at(int i, int j);
    const float& at(int i, int j) const;

    // For 3D convenience
    float& at(int i, int j, int k);
    const float& at(int i, int j, int k) const;

    const std::array<int, Dim>& shape() const { return shape_; }
    int size(int dim) const { return shape_[dim]; }
    std::size_t total_size() const;

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    void fill(float value);
    void fill_random(float low, float high, unsigned seed);

    // Place a circular/spherical blob at center
    void place_blob(const std::array<int, Dim>& center, int radius, float value);

    // Total mass (sum of all states)
    double total_mass() const;

    // Clip all values to [0,1]
    void clip();

private:
    std::size_t linearize(const std::array<int, Dim>& idx) const;
    std::vector<float> data_;
    std::array<int, Dim> shape_{};
};

// Explicit instantiation declarations
extern template class Grid<2>;
extern template class Grid<3>;
