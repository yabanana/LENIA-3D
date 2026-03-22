#include <gtest/gtest.h>
#include "core/grid.h"

#include <cmath>
#include <numeric>

// ===========================================================================
// Grid<2> Tests
// ===========================================================================

TEST(Grid2D, ConstructionCreatesZeroFilledGrid) {
    Grid<2> grid(std::array<int, 2>{16, 16});

    EXPECT_EQ(grid.size(0), 16);
    EXPECT_EQ(grid.size(1), 16);
    EXPECT_EQ(grid.total_size(), 256u);

    for (std::size_t i = 0; i < grid.total_size(); ++i) {
        EXPECT_FLOAT_EQ(grid.data()[i], 0.0f);
    }
}

TEST(Grid2D, ShapeReturnsCorrectDimensions) {
    Grid<2> grid(std::array<int, 2>{32, 64});

    const auto& shape = grid.shape();
    EXPECT_EQ(shape[0], 32);
    EXPECT_EQ(shape[1], 64);
    EXPECT_EQ(grid.total_size(), 32u * 64u);
}

TEST(Grid2D, ElementAccessWithAt) {
    Grid<2> grid(std::array<int, 2>{8, 8});

    grid.at(3, 5) = 0.75f;
    EXPECT_FLOAT_EQ(grid.at(3, 5), 0.75f);

    // Verify other cells are still zero
    EXPECT_FLOAT_EQ(grid.at(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(grid.at(7, 7), 0.0f);
}

TEST(Grid2D, ElementAccessWithOperatorParens) {
    Grid<2> grid(std::array<int, 2>{8, 8});

    std::array<int, 2> idx = {2, 4};
    grid(idx) = 0.5f;

    EXPECT_FLOAT_EQ(grid(idx), 0.5f);
    EXPECT_FLOAT_EQ(grid.at(2, 4), 0.5f);
}

TEST(Grid2D, FillSetsAllCells) {
    Grid<2> grid(std::array<int, 2>{10, 10});

    grid.fill(0.42f);

    for (std::size_t i = 0; i < grid.total_size(); ++i) {
        EXPECT_FLOAT_EQ(grid.data()[i], 0.42f);
    }
}

TEST(Grid2D, FillRandomProducesValuesInRange) {
    Grid<2> grid(std::array<int, 2>{32, 32});

    grid.fill_random(0.2f, 0.8f, 12345);

    bool has_nonzero = false;
    for (std::size_t i = 0; i < grid.total_size(); ++i) {
        EXPECT_GE(grid.data()[i], 0.2f);
        EXPECT_LE(grid.data()[i], 0.8f);
        if (grid.data()[i] != 0.0f) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(Grid2D, FillRandomIsDeterministicWithSameSeed) {
    Grid<2> g1(std::array<int, 2>{16, 16});
    Grid<2> g2(std::array<int, 2>{16, 16});

    g1.fill_random(0.0f, 1.0f, 99);
    g2.fill_random(0.0f, 1.0f, 99);

    for (std::size_t i = 0; i < g1.total_size(); ++i) {
        EXPECT_FLOAT_EQ(g1.data()[i], g2.data()[i]);
    }
}

TEST(Grid2D, TotalMassComputesSum) {
    Grid<2> grid(std::array<int, 2>{4, 4});
    grid.fill(0.25f);

    double expected = 0.25 * 16.0;
    EXPECT_NEAR(grid.total_mass(), expected, 1e-6);
}

TEST(Grid2D, TotalMassZeroForEmptyGrid) {
    Grid<2> grid(std::array<int, 2>{8, 8});
    EXPECT_DOUBLE_EQ(grid.total_mass(), 0.0);
}

TEST(Grid2D, ClipClampsValues) {
    Grid<2> grid(std::array<int, 2>{4, 4});

    grid.at(0, 0) = -0.5f;
    grid.at(0, 1) = 0.5f;
    grid.at(1, 0) = 1.5f;
    grid.at(1, 1) = 2.0f;
    grid.at(2, 0) = -100.0f;

    grid.clip();

    EXPECT_FLOAT_EQ(grid.at(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(grid.at(0, 1), 0.5f);
    EXPECT_FLOAT_EQ(grid.at(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(grid.at(1, 1), 1.0f);
    EXPECT_FLOAT_EQ(grid.at(2, 0), 0.0f);
}

TEST(Grid2D, PlaceBlobCreatesCircularPattern) {
    Grid<2> grid(std::array<int, 2>{32, 32});
    std::array<int, 2> center = {16, 16};
    int radius = 3;
    float value = 0.8f;

    grid.place_blob(center, radius, value);

    // Center should be set
    EXPECT_FLOAT_EQ(grid.at(16, 16), value);

    // Points within radius should be set
    EXPECT_FLOAT_EQ(grid.at(16, 17), value);
    EXPECT_FLOAT_EQ(grid.at(15, 16), value);

    // Points far outside radius should remain zero
    EXPECT_FLOAT_EQ(grid.at(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(grid.at(25, 25), 0.0f);

    // Total mass should be positive
    EXPECT_GT(grid.total_mass(), 0.0);
}

TEST(Grid2D, PlaceBlobWrapsAround) {
    Grid<2> grid(std::array<int, 2>{16, 16});
    std::array<int, 2> center = {0, 0};
    int radius = 2;

    grid.place_blob(center, radius, 1.0f);

    // Should wrap to opposite edges
    EXPECT_FLOAT_EQ(grid.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(grid.at(15, 0), 1.0f);  // -1 wraps to 15
    EXPECT_FLOAT_EQ(grid.at(0, 15), 1.0f);  // -1 wraps to 15
}

TEST(Grid2D, LinearizationIsRowMajor) {
    Grid<2> grid(std::array<int, 2>{4, 8});

    // In row-major order, at(i,j) should correspond to data[i*cols + j]
    grid.at(2, 3) = 1.0f;
    EXPECT_FLOAT_EQ(grid.data()[2 * 8 + 3], 1.0f);

    grid.at(0, 0) = 2.0f;
    EXPECT_FLOAT_EQ(grid.data()[0], 2.0f);

    grid.at(3, 7) = 3.0f;
    EXPECT_FLOAT_EQ(grid.data()[3 * 8 + 7], 3.0f);
}

TEST(Grid2D, DataPointerAllowsDirectAccess) {
    Grid<2> grid(std::array<int, 2>{4, 4});

    float* ptr = grid.data();
    ptr[5] = 0.99f;

    EXPECT_FLOAT_EQ(grid.at(1, 1), 0.99f);
}

// ===========================================================================
// Grid<3> Tests
// ===========================================================================

TEST(Grid3D, ConstructionCreatesZeroFilledGrid) {
    Grid<3> grid(std::array<int, 3>{8, 8, 8});

    EXPECT_EQ(grid.size(0), 8);
    EXPECT_EQ(grid.size(1), 8);
    EXPECT_EQ(grid.size(2), 8);
    EXPECT_EQ(grid.total_size(), 512u);

    for (std::size_t i = 0; i < grid.total_size(); ++i) {
        EXPECT_FLOAT_EQ(grid.data()[i], 0.0f);
    }
}

TEST(Grid3D, ElementAccessWithAt) {
    Grid<3> grid(std::array<int, 3>{4, 4, 4});

    grid.at(1, 2, 3) = 0.33f;
    EXPECT_FLOAT_EQ(grid.at(1, 2, 3), 0.33f);

    // Other cells remain zero
    EXPECT_FLOAT_EQ(grid.at(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(grid.at(3, 3, 3), 0.0f);
}

TEST(Grid3D, FillAndMass) {
    Grid<3> grid(std::array<int, 3>{4, 4, 4});
    grid.fill(0.5f);

    double expected = 0.5 * 64.0;
    EXPECT_NEAR(grid.total_mass(), expected, 1e-5);
}

TEST(Grid3D, PlaceBlobCreatesSphericalPattern) {
    Grid<3> grid(std::array<int, 3>{32, 32, 32});
    std::array<int, 3> center = {16, 16, 16};
    int radius = 4;
    float value = 0.7f;

    grid.place_blob(center, radius, value);

    // Center should be set
    EXPECT_FLOAT_EQ(grid.at(16, 16, 16), value);

    // Adjacent cells should be set
    EXPECT_FLOAT_EQ(grid.at(16, 16, 17), value);

    // Far cells should remain zero
    EXPECT_FLOAT_EQ(grid.at(0, 0, 0), 0.0f);

    // Total mass should be positive
    EXPECT_GT(grid.total_mass(), 0.0);
}

TEST(Grid3D, ClipClampsValues) {
    Grid<3> grid(std::array<int, 3>{2, 2, 2});
    grid.at(0, 0, 0) = -1.0f;
    grid.at(0, 0, 1) = 0.5f;
    grid.at(1, 1, 1) = 3.0f;

    grid.clip();

    EXPECT_FLOAT_EQ(grid.at(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(grid.at(0, 0, 1), 0.5f);
    EXPECT_FLOAT_EQ(grid.at(1, 1, 1), 1.0f);
}

TEST(Grid3D, LinearizationIsRowMajor) {
    Grid<3> grid(std::array<int, 3>{2, 3, 4});

    // Row-major: at(i,j,k) -> data[i*3*4 + j*4 + k]
    grid.at(1, 2, 3) = 5.0f;
    EXPECT_FLOAT_EQ(grid.data()[1 * 3 * 4 + 2 * 4 + 3], 5.0f);
}

TEST(Grid3D, FillRandomProducesValuesInRange) {
    Grid<3> grid(std::array<int, 3>{8, 8, 8});
    grid.fill_random(0.1f, 0.9f, 42);

    for (std::size_t i = 0; i < grid.total_size(); ++i) {
        EXPECT_GE(grid.data()[i], 0.1f);
        EXPECT_LE(grid.data()[i], 0.9f);
    }
}
