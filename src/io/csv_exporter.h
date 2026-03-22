#pragma once

#include <string>
#include <utility>
#include <vector>

// Export 2D grid state to CSV (one value per cell, rows = grid rows)
void export_grid_2d_csv(const float* data, int rows, int cols,
                        const std::string& filename);

// Export 3D grid state to CSV (one slice per section, separated by blank line)
void export_grid_3d_csv(const float* data, int d, int r, int c,
                        const std::string& filename);

// Export time-series metrics (iteration, mass, etc.)
void export_metrics_csv(const std::string& filename,
                        const std::vector<std::pair<int, double>>& data,
                        const std::string& header = "iteration,mass");
