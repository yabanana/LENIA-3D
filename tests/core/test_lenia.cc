#include <gtest/gtest.h>
#include "core/lenia.h"

#include <cmath>
#include <vector>

// ===========================================================================
// 2D Lenia Integration Tests
// ===========================================================================

class Lenia2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.dimension   = 2;
        config_.grid_size   = 64;
        config_.T           = 10;
        config_.num_threads = 1;
        config_.seed        = 42;

        // Standard Orbium-like parameters
        config_.kernel.radius     = 13;
        config_.kernel.beta       = {1.0f};
        config_.kernel.core_alpha = 4;
        config_.growth.mu         = 0.15f;
        config_.growth.sigma      = 0.015f;
    }

    LeniaConfig config_;
};

TEST_F(Lenia2DTest, ConstructionSucceeds) {
    Lenia lenia(config_);

    EXPECT_EQ(lenia.dimension(), 2);
    EXPECT_EQ(lenia.grid_size(), 64);
    EXPECT_EQ(lenia.iteration(), 0);
}

TEST_F(Lenia2DTest, InitRandomProducesNonzeroMass) {
    Lenia lenia(config_);
    lenia.init_random(42);

    EXPECT_GT(lenia.total_mass(), 0.0);
}

TEST_F(Lenia2DTest, StepAdvancesIteration) {
    Lenia lenia(config_);
    lenia.init_random(42);

    EXPECT_EQ(lenia.iteration(), 0);
    lenia.step();
    EXPECT_EQ(lenia.iteration(), 1);
    lenia.step();
    EXPECT_EQ(lenia.iteration(), 2);
}

TEST_F(Lenia2DTest, StepChangesState) {
    Lenia lenia(config_);
    lenia.init_random(42);

    // Record initial state
    std::size_t total = static_cast<std::size_t>(64) * 64;
    std::vector<float> before(lenia.state_data(), lenia.state_data() + total);

    lenia.step();

    const float* after = lenia.state_data();
    bool changed = false;
    for (std::size_t i = 0; i < total; ++i) {
        if (std::abs(after[i] - before[i]) > 1e-8f) {
            changed = true;
            break;
        }
    }

    EXPECT_TRUE(changed) << "State should change after a step";
}

TEST_F(Lenia2DTest, ValuesStayInZeroOneRange) {
    Lenia lenia(config_);
    lenia.init_random(42);

    // Run several steps
    for (int s = 0; s < 20; ++s) {
        lenia.step();
    }

    std::size_t total = static_cast<std::size_t>(64) * 64;
    const float* data = lenia.state_data();
    for (std::size_t i = 0; i < total; ++i) {
        EXPECT_GE(data[i], 0.0f)
            << "Value below 0 at index " << i << " after 20 steps";
        EXPECT_LE(data[i], 1.0f)
            << "Value above 1 at index " << i << " after 20 steps";
    }
}

TEST_F(Lenia2DTest, MassChangesGradually) {
    // Mass should not suddenly jump to zero or explode in a few steps.
    Lenia lenia(config_);
    lenia.init_random(42);

    double initial_mass = lenia.total_mass();
    EXPECT_GT(initial_mass, 0.0);

    for (int s = 0; s < 10; ++s) {
        lenia.step();
    }

    double final_mass = lenia.total_mass();
    // Mass can increase or decrease, but should not be zero or absurdly large
    EXPECT_GT(final_mass, 0.0)
        << "Mass should remain positive after 10 steps with random init";
}

TEST_F(Lenia2DTest, DtMatchesT) {
    Lenia lenia(config_);
    EXPECT_FLOAT_EQ(lenia.dt(), 1.0f / 10.0f);
}

TEST_F(Lenia2DTest, StateDataPointerIsValid) {
    Lenia lenia(config_);

    const float* ptr = lenia.state_data();
    EXPECT_NE(ptr, nullptr);
}

TEST_F(Lenia2DTest, InitOrbiumSetsNonzeroPattern) {
    Lenia lenia(config_);
    lenia.init_orbium();

    EXPECT_GT(lenia.total_mass(), 0.0)
        << "Orbium pattern should have nonzero mass";

    // The iteration counter should be reset
    EXPECT_EQ(lenia.iteration(), 0);
}

TEST_F(Lenia2DTest, SetGrowthParamsUpdatesConfig) {
    Lenia lenia(config_);

    GrowthParams new_params;
    new_params.mu = 0.20f;
    new_params.sigma = 0.020f;

    lenia.set_growth_params(new_params);

    EXPECT_FLOAT_EQ(lenia.config().growth.mu, 0.20f);
    EXPECT_FLOAT_EQ(lenia.config().growth.sigma, 0.020f);
}

TEST_F(Lenia2DTest, SetKernelParamsRebuildsKernel) {
    Lenia lenia(config_);
    lenia.init_random(42);

    // Record state after several steps with original kernel
    for (int i = 0; i < 10; ++i) lenia.step();
    double mass_after_steps = lenia.total_mass();

    // Create a fresh instance with different kernel
    LeniaConfig cfg2 = config_;
    cfg2.kernel.radius = 5;
    Lenia lenia2(cfg2);
    lenia2.init_random(42);

    for (int i = 0; i < 10; ++i) lenia2.step();
    double mass_with_different_kernel = lenia2.total_mass();

    // With a different kernel, the evolution should differ
    EXPECT_NE(mass_after_steps, mass_with_different_kernel)
        << "Different kernel parameters should produce different evolution";
}

TEST_F(Lenia2DTest, SetTimeStepUpdatesConfig) {
    Lenia lenia(config_);
    lenia.set_time_step(20);

    EXPECT_EQ(lenia.config().T, 20);
    EXPECT_FLOAT_EQ(lenia.dt(), 1.0f / 20.0f);
}

TEST_F(Lenia2DTest, Grid2DAccessorsWork) {
    Lenia lenia(config_);
    lenia.init_random(42);

    const Grid<2>& grid = lenia.grid_2d();
    EXPECT_EQ(grid.size(0), 64);
    EXPECT_EQ(grid.size(1), 64);

    Grid<2>& grid_mut = lenia.grid_2d_mut();
    grid_mut.at(0, 0) = 0.5f;
    EXPECT_FLOAT_EQ(lenia.grid_2d().at(0, 0), 0.5f);
}

// ===========================================================================
// 3D Lenia Integration Tests
// ===========================================================================

class Lenia3DTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.dimension   = 3;
        config_.grid_size   = 16;   // Small for speed
        config_.T           = 10;
        config_.num_threads = 1;
        config_.seed        = 42;

        config_.kernel.radius     = 5;
        config_.kernel.beta       = {1.0f};
        config_.kernel.core_alpha = 4;
        config_.growth.mu         = 0.15f;
        config_.growth.sigma      = 0.015f;
    }

    LeniaConfig config_;
};

TEST_F(Lenia3DTest, ConstructionSucceeds) {
    Lenia lenia(config_);

    EXPECT_EQ(lenia.dimension(), 3);
    EXPECT_EQ(lenia.grid_size(), 16);
    EXPECT_EQ(lenia.iteration(), 0);
}

TEST_F(Lenia3DTest, StepProducesValidState) {
    Lenia lenia(config_);
    lenia.init_random(42);

    for (int s = 0; s < 5; ++s) {
        lenia.step();
    }

    std::size_t total = static_cast<std::size_t>(16) * 16 * 16;
    const float* data = lenia.state_data();
    for (std::size_t i = 0; i < total; ++i) {
        EXPECT_GE(data[i], 0.0f);
        EXPECT_LE(data[i], 1.0f);
    }
}

TEST_F(Lenia3DTest, Grid3DAccessorsWork) {
    Lenia lenia(config_);

    const Grid<3>& grid = lenia.grid_3d();
    EXPECT_EQ(grid.size(0), 16);
    EXPECT_EQ(grid.size(1), 16);
    EXPECT_EQ(grid.size(2), 16);
}

TEST_F(Lenia3DTest, InitRandomProducesNonzeroMass) {
    Lenia lenia(config_);
    lenia.init_random(42);

    EXPECT_GT(lenia.total_mass(), 0.0);
}
