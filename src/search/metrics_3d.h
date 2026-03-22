#pragma once

#include "core/grid.h"

#include <array>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Snapshot of metrics computed at a single simulation step.
// ---------------------------------------------------------------------------
struct MetricsSnapshot {
    int step;
    double mass;
    double mass_ratio;                // mass / initial_mass
    std::array<double, 3> com;        // center of mass (x, y, z)
    double com_displacement;          // distance from initial CoM
    double radius_of_gyration;        // Rg = sqrt(sum(m_i*|r_i - r_cm|^2) / M)
    double compactness;               // mass / (4/3 pi Rg^3)
    double spatial_entropy;           // Shannon entropy, normalized [0,1]
    std::array<int, 3> bbox_min;      // bounding box of cells > threshold
    std::array<int, 3> bbox_max;
    double surface_volume_ratio;      // surface cells / interior cells
};

// ---------------------------------------------------------------------------
// Classification of a completed run.
// ---------------------------------------------------------------------------
enum class PatternClass {
    Extinct,   // mass died out
    Filled,    // mass filled the domain
    Stable,    // localized, low variance
    Pulsing,   // oscillating mass
    Gliding,   // sustained CoM displacement
    Chaotic,   // high entropy, high variance
    Unknown
};

const char* pattern_class_name(PatternClass c);

// ---------------------------------------------------------------------------
// Summary statistics for one parameter-sweep run.
// ---------------------------------------------------------------------------
struct RunSummary {
    float sigma;
    float mu;
    int radius;
    int T;
    std::string pattern;

    double final_mass_ratio;
    double mass_variance;
    double mean_com_speed;       // cells per sample interval
    double final_rg;
    double final_compactness;
    double final_entropy;
    double final_sv_ratio;

    double dominant_frequency;   // in cycles/sample, 0 if none
    double oscillation_amplitude;

    std::array<int, 3> final_bbox_size;  // dx, dy, dz

    PatternClass classification;
    bool early_terminated;
    int steps_run;
};

// ---------------------------------------------------------------------------
// Individual metric computations (all operate on const Grid<3>&).
// ---------------------------------------------------------------------------

// Center of mass — OpenMP reduction over the flat grid array.
// Returns {com_x, com_y, com_z} in grid coordinates.
std::array<double, 3> compute_center_of_mass(const Grid<3>& grid);

// Radius of gyration around a given center of mass.
double compute_radius_of_gyration(const Grid<3>& grid,
                                  const std::array<double, 3>& com);

// Compactness = total_mass / (4/3 pi Rg^3).
double compute_compactness(double total_mass, double rg);

// Shannon entropy of the state distribution (20 bins on (0,1]).
// Cells with value < 1e-6 are ignored (dead cells).
// Normalized to [0,1] by dividing by log2(num_bins).
double compute_spatial_entropy(const Grid<3>& grid, int num_bins = 20);

// Axis-aligned bounding box of cells with value > threshold.
// Returns false if no cells exceed the threshold.
bool compute_bounding_box(const Grid<3>& grid, float threshold,
                          std::array<int, 3>& bb_min,
                          std::array<int, 3>& bb_max);

// Surface-to-volume ratio. A cell > threshold is "surface" if at least one
// of its 6 face-neighbours is < threshold (periodic boundary).
// Returns surface_count / interior_count, or 0 if no interior.
double compute_surface_volume_ratio(const Grid<3>& grid, float threshold);

// Compute all metrics for one snapshot.
MetricsSnapshot compute_all_metrics(const Grid<3>& grid, int step,
                                    double initial_mass,
                                    const std::array<double, 3>& initial_com,
                                    float threshold = 0.01f);

// ---------------------------------------------------------------------------
// Time-series analysis and classification.
// ---------------------------------------------------------------------------

// Detect dominant oscillation in the mass time series via autocorrelation.
// Returns {dominant_period, amplitude}. Period=0 if no periodicity found.
std::pair<double, double> detect_oscillation(
    const std::vector<MetricsSnapshot>& history);

// Classify a run from its snapshot history.
RunSummary classify_run(const std::vector<MetricsSnapshot>& history,
                        float sigma, float mu, int radius, int T,
                        const std::string& pattern,
                        bool early_terminated, int steps_run);
