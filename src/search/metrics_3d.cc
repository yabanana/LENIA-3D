#include "search/metrics_3d.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// pattern_class_name()
// ---------------------------------------------------------------------------
const char* pattern_class_name(PatternClass c) {
    switch (c) {
        case PatternClass::Extinct:  return "Extinct";
        case PatternClass::Filled:   return "Filled";
        case PatternClass::Stable:   return "Stable";
        case PatternClass::Pulsing:  return "Pulsing";
        case PatternClass::Gliding:  return "Gliding";
        case PatternClass::Chaotic:  return "Chaotic";
        case PatternClass::Unknown:  return "Unknown";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// compute_center_of_mass()
// ---------------------------------------------------------------------------
std::array<double, 3> compute_center_of_mass(const Grid<3>& grid) {
    const int N0 = grid.size(0);
    const int N1 = grid.size(1);
    const int N2 = grid.size(2);
    const float* data = grid.data();

    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0, total = 0.0;

    #pragma omp parallel for reduction(+:sum_x,sum_y,sum_z,total) schedule(static)
    for (int k = 0; k < N0; ++k) {
        for (int i = 0; i < N1; ++i) {
            for (int j = 0; j < N2; ++j) {
                std::size_t idx = static_cast<std::size_t>(k) * N1 * N2
                                + static_cast<std::size_t>(i) * N2 + j;
                double val = static_cast<double>(data[idx]);
                if (val > 1e-6) {
                    sum_x += val * j;  // x = column
                    sum_y += val * i;  // y = row
                    sum_z += val * k;  // z = depth
                    total += val;
                }
            }
        }
    }

    if (total < 1e-12) {
        double c = static_cast<double>(N2) / 2.0;
        return {c, c, c};
    }

    return {sum_x / total, sum_y / total, sum_z / total};
}

// ---------------------------------------------------------------------------
// compute_radius_of_gyration()
// ---------------------------------------------------------------------------
double compute_radius_of_gyration(const Grid<3>& grid,
                                  const std::array<double, 3>& com) {
    const int N0 = grid.size(0);
    const int N1 = grid.size(1);
    const int N2 = grid.size(2);
    const float* data = grid.data();

    double sum_r2 = 0.0, total = 0.0;

    #pragma omp parallel for reduction(+:sum_r2,total) schedule(static)
    for (int k = 0; k < N0; ++k) {
        for (int i = 0; i < N1; ++i) {
            for (int j = 0; j < N2; ++j) {
                std::size_t idx = static_cast<std::size_t>(k) * N1 * N2
                                + static_cast<std::size_t>(i) * N2 + j;
                double val = static_cast<double>(data[idx]);
                if (val > 1e-6) {
                    double dx = j - com[0];
                    double dy = i - com[1];
                    double dz = k - com[2];
                    sum_r2 += val * (dx * dx + dy * dy + dz * dz);
                    total += val;
                }
            }
        }
    }

    if (total < 1e-12) return 0.0;
    return std::sqrt(sum_r2 / total);
}

// ---------------------------------------------------------------------------
// compute_compactness()
// ---------------------------------------------------------------------------
double compute_compactness(double total_mass, double rg) {
    if (rg < 1e-6) return 0.0;
    constexpr double four_thirds_pi = 4.0 / 3.0 * M_PI;
    double volume = four_thirds_pi * rg * rg * rg;
    return total_mass / volume;
}

// ---------------------------------------------------------------------------
// compute_spatial_entropy()
// ---------------------------------------------------------------------------
double compute_spatial_entropy(const Grid<3>& grid, int num_bins) {
    const float* data = grid.data();
    const std::size_t n = grid.total_size();

    std::vector<long long> bins(num_bins, 0);
    long long active_count = 0;

    // Build histogram — cells with val < 1e-6 are considered dead
    for (std::size_t idx = 0; idx < n; ++idx) {
        float val = data[idx];
        if (val < 1e-6f) continue;
        int b = static_cast<int>(val * num_bins);
        if (b >= num_bins) b = num_bins - 1;
        bins[b]++;
        active_count++;
    }

    if (active_count == 0) return 0.0;

    double H = 0.0;
    double inv_total = 1.0 / static_cast<double>(active_count);
    for (int b = 0; b < num_bins; ++b) {
        if (bins[b] == 0) continue;
        double p = static_cast<double>(bins[b]) * inv_total;
        H -= p * std::log2(p);
    }

    // Normalize to [0, 1]
    double max_H = std::log2(static_cast<double>(num_bins));
    return (max_H > 0.0) ? H / max_H : 0.0;
}

// ---------------------------------------------------------------------------
// compute_bounding_box()
// ---------------------------------------------------------------------------
bool compute_bounding_box(const Grid<3>& grid, float threshold,
                          std::array<int, 3>& bb_min,
                          std::array<int, 3>& bb_max) {
    const int N0 = grid.size(0);
    const int N1 = grid.size(1);
    const int N2 = grid.size(2);
    const float* data = grid.data();

    int min_k = N0, min_i = N1, min_j = N2;
    int max_k = -1, max_i = -1, max_j = -1;

    for (int k = 0; k < N0; ++k) {
        for (int i = 0; i < N1; ++i) {
            for (int j = 0; j < N2; ++j) {
                std::size_t idx = static_cast<std::size_t>(k) * N1 * N2
                                + static_cast<std::size_t>(i) * N2 + j;
                if (data[idx] > threshold) {
                    if (k < min_k) min_k = k;
                    if (i < min_i) min_i = i;
                    if (j < min_j) min_j = j;
                    if (k > max_k) max_k = k;
                    if (i > max_i) max_i = i;
                    if (j > max_j) max_j = j;
                }
            }
        }
    }

    if (max_k < 0) return false;

    bb_min = {min_j, min_i, min_k};  // x, y, z
    bb_max = {max_j, max_i, max_k};
    return true;
}

// ---------------------------------------------------------------------------
// compute_surface_volume_ratio()
// ---------------------------------------------------------------------------
double compute_surface_volume_ratio(const Grid<3>& grid, float threshold) {
    const int N0 = grid.size(0);
    const int N1 = grid.size(1);
    const int N2 = grid.size(2);
    const float* data = grid.data();

    long long surface_count = 0;
    long long interior_count = 0;

    #pragma omp parallel for reduction(+:surface_count,interior_count) schedule(static)
    for (int k = 0; k < N0; ++k) {
        for (int i = 0; i < N1; ++i) {
            for (int j = 0; j < N2; ++j) {
                std::size_t idx = static_cast<std::size_t>(k) * N1 * N2
                                + static_cast<std::size_t>(i) * N2 + j;
                if (data[idx] <= threshold) continue;

                // Check 6 face-neighbours with periodic boundary
                bool is_surface = false;

                // j-1, j+1
                int jm = (j - 1 + N2) % N2;
                int jp = (j + 1) % N2;
                // i-1, i+1
                int im = (i - 1 + N1) % N1;
                int ip = (i + 1) % N1;
                // k-1, k+1
                int km = (k - 1 + N0) % N0;
                int kp = (k + 1) % N0;

                auto at = [&](int kk, int ii, int jj) -> float {
                    return data[static_cast<std::size_t>(kk) * N1 * N2
                              + static_cast<std::size_t>(ii) * N2 + jj];
                };

                if (at(k, i, jm) <= threshold || at(k, i, jp) <= threshold ||
                    at(k, im, j) <= threshold || at(k, ip, j) <= threshold ||
                    at(km, i, j) <= threshold || at(kp, i, j) <= threshold) {
                    is_surface = true;
                }

                if (is_surface) {
                    surface_count++;
                } else {
                    interior_count++;
                }
            }
        }
    }

    if (interior_count == 0) {
        return (surface_count > 0) ? static_cast<double>(surface_count) : 0.0;
    }
    return static_cast<double>(surface_count) / static_cast<double>(interior_count);
}

// ---------------------------------------------------------------------------
// compute_all_metrics()
// ---------------------------------------------------------------------------
MetricsSnapshot compute_all_metrics(const Grid<3>& grid, int step,
                                    double initial_mass,
                                    const std::array<double, 3>& initial_com,
                                    float threshold) {
    MetricsSnapshot snap;
    snap.step = step;
    snap.mass = grid.total_mass();
    snap.mass_ratio = (initial_mass > 1e-12) ? snap.mass / initial_mass : 0.0;

    snap.com = compute_center_of_mass(grid);

    double dx = snap.com[0] - initial_com[0];
    double dy = snap.com[1] - initial_com[1];
    double dz = snap.com[2] - initial_com[2];
    snap.com_displacement = std::sqrt(dx * dx + dy * dy + dz * dz);

    snap.radius_of_gyration = compute_radius_of_gyration(grid, snap.com);
    snap.compactness = compute_compactness(snap.mass, snap.radius_of_gyration);
    snap.spatial_entropy = compute_spatial_entropy(grid);

    if (!compute_bounding_box(grid, threshold, snap.bbox_min, snap.bbox_max)) {
        snap.bbox_min = {0, 0, 0};
        snap.bbox_max = {0, 0, 0};
    }

    snap.surface_volume_ratio = compute_surface_volume_ratio(grid, threshold);

    return snap;
}

// ---------------------------------------------------------------------------
// detect_oscillation()
// ---------------------------------------------------------------------------
std::pair<double, double> detect_oscillation(
    const std::vector<MetricsSnapshot>& history) {
    if (history.size() < 6) return {0.0, 0.0};

    const std::size_t n = history.size();

    // Extract mass series
    std::vector<double> mass(n);
    for (std::size_t i = 0; i < n; ++i) {
        mass[i] = history[i].mass_ratio;
    }

    // Compute mean
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean += mass[i];
    mean /= static_cast<double>(n);

    // Autocorrelation R(tau) for tau = 0..n/2
    std::size_t max_lag = n / 2;
    std::vector<double> acorr(max_lag + 1, 0.0);

    for (std::size_t tau = 0; tau <= max_lag; ++tau) {
        double sum = 0.0;
        std::size_t count = n - tau;
        for (std::size_t t = 0; t < count; ++t) {
            sum += (mass[t] - mean) * (mass[t + tau] - mean);
        }
        acorr[tau] = sum / static_cast<double>(count);
    }

    if (std::abs(acorr[0]) < 1e-12) return {0.0, 0.0};

    // Find first peak after tau > 2 (skip trivial near-zero-lag peak)
    double best_peak = 0.0;
    std::size_t best_tau = 0;
    for (std::size_t tau = 3; tau < max_lag; ++tau) {
        if (acorr[tau] > acorr[tau - 1] && acorr[tau] > acorr[tau + 1]) {
            if (acorr[tau] > best_peak) {
                best_peak = acorr[tau];
                best_tau = tau;
            }
            break;  // take first peak
        }
    }

    double period = 0.0;
    if (best_tau > 0 && best_peak / acorr[0] > 0.3) {
        period = static_cast<double>(best_tau);
    }

    // Amplitude = max - min of mass ratio
    double min_m = *std::min_element(mass.begin(), mass.end());
    double max_m = *std::max_element(mass.begin(), mass.end());
    double amplitude = max_m - min_m;

    return {period, amplitude};
}

// ---------------------------------------------------------------------------
// classify_run()
// ---------------------------------------------------------------------------
RunSummary classify_run(const std::vector<MetricsSnapshot>& history,
                        float sigma, float mu, int radius, int T,
                        const std::string& pattern,
                        bool early_terminated, int steps_run) {
    RunSummary summary;
    summary.sigma = sigma;
    summary.mu = mu;
    summary.radius = radius;
    summary.T = T;
    summary.pattern = pattern;
    summary.early_terminated = early_terminated;
    summary.steps_run = steps_run;

    if (history.empty()) {
        summary.final_mass_ratio = 0.0;
        summary.mass_variance = 0.0;
        summary.mean_com_speed = 0.0;
        summary.final_rg = 0.0;
        summary.final_compactness = 0.0;
        summary.final_entropy = 0.0;
        summary.final_sv_ratio = 0.0;
        summary.dominant_frequency = 0.0;
        summary.oscillation_amplitude = 0.0;
        summary.final_bbox_size = {0, 0, 0};
        summary.classification = PatternClass::Unknown;
        return summary;
    }

    const auto& last = history.back();
    summary.final_mass_ratio = last.mass_ratio;
    summary.final_rg = last.radius_of_gyration;
    summary.final_compactness = last.compactness;
    summary.final_entropy = last.spatial_entropy;
    summary.final_sv_ratio = last.surface_volume_ratio;

    // Bounding box size
    summary.final_bbox_size = {
        last.bbox_max[0] - last.bbox_min[0] + 1,
        last.bbox_max[1] - last.bbox_min[1] + 1,
        last.bbox_max[2] - last.bbox_min[2] + 1
    };

    // Mass variance
    const std::size_t n = history.size();
    double mean_mass = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean_mass += history[i].mass_ratio;
    mean_mass /= static_cast<double>(n);

    double var = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = history[i].mass_ratio - mean_mass;
        var += d * d;
    }
    summary.mass_variance = var / static_cast<double>(n);

    // Mean CoM speed (displacement between consecutive samples)
    double total_speed = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        double dx = history[i].com[0] - history[i - 1].com[0];
        double dy = history[i].com[1] - history[i - 1].com[1];
        double dz = history[i].com[2] - history[i - 1].com[2];
        total_speed += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    summary.mean_com_speed = (n > 1) ? total_speed / static_cast<double>(n - 1) : 0.0;

    // Oscillation analysis
    auto [period, amplitude] = detect_oscillation(history);
    summary.dominant_frequency = (period > 0.0) ? 1.0 / period : 0.0;
    summary.oscillation_amplitude = amplitude;

    // --- Classification decision tree ---
    // Priority 1: Extinct
    if (summary.final_mass_ratio < 0.05) {
        summary.classification = PatternClass::Extinct;
    }
    // Priority 2: Filled
    else if (summary.final_mass_ratio > 0.80 && summary.final_compactness < 0.1) {
        summary.classification = PatternClass::Filled;
    }
    // Priority 3: Gliding
    else if (summary.mean_com_speed > 0.5) {
        summary.classification = PatternClass::Gliding;
    }
    // Priority 4: Pulsing
    else if (amplitude > 0.05 && summary.mass_variance > 1e-4) {
        summary.classification = PatternClass::Pulsing;
    }
    // Priority 5: Stable
    else if (summary.mass_variance < 1e-5 &&
             summary.final_mass_ratio > 0.05 &&
             summary.final_mass_ratio < 0.80) {
        summary.classification = PatternClass::Stable;
    }
    // Priority 6: Chaotic
    else if (summary.final_entropy > 0.7 && summary.mass_variance > 1e-3) {
        summary.classification = PatternClass::Chaotic;
    }
    // Priority 7: Unknown
    else {
        summary.classification = PatternClass::Unknown;
    }

    return summary;
}
