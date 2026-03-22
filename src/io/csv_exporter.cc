#include "io/csv_exporter.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

// ---------------------------------------------------------------------------
// export_grid_2d_csv
// Writes a 2D grid to a CSV file. Each row of the grid becomes one line
// in the CSV, with float values separated by commas and printed with 6
// decimal places of precision.
// ---------------------------------------------------------------------------
void export_grid_2d_csv(const float* data, int rows, int cols,
                        const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    out << std::fixed << std::setprecision(6);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (j > 0) {
                out << ',';
            }
            out << data[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols)
                        + static_cast<std::size_t>(j)];
        }
        out << '\n';
    }
}

// ---------------------------------------------------------------------------
// export_grid_3d_csv
// Writes a 3D grid to a CSV file. Each depth slice is written as a block
// of rows x cols values. Slices are separated by a blank line.
// A comment line "# slice=N" precedes each slice for readability.
// ---------------------------------------------------------------------------
void export_grid_3d_csv(const float* data, int d, int r, int c,
                        const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    out << std::fixed << std::setprecision(6);

    std::size_t slice_size = static_cast<std::size_t>(r) * static_cast<std::size_t>(c);

    for (int depth = 0; depth < d; ++depth) {
        out << "# slice=" << depth << '\n';
        const float* slice = data + static_cast<std::size_t>(depth) * slice_size;

        for (int row = 0; row < r; ++row) {
            for (int col = 0; col < c; ++col) {
                if (col > 0) {
                    out << ',';
                }
                out << slice[static_cast<std::size_t>(row)
                             * static_cast<std::size_t>(c)
                             + static_cast<std::size_t>(col)];
            }
            out << '\n';
        }

        // Blank line between slices (except after the last one)
        if (depth < d - 1) {
            out << '\n';
        }
    }
}

// ---------------------------------------------------------------------------
// export_metrics_csv
// Writes time-series data as a two-column CSV with a header line.
// Each pair is (iteration, metric_value).
// ---------------------------------------------------------------------------
void export_metrics_csv(const std::string& filename,
                        const std::vector<std::pair<int, double>>& data,
                        const std::string& header) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    out << header << '\n';
    out << std::fixed << std::setprecision(6);

    for (const auto& entry : data) {
        out << entry.first << ',' << entry.second << '\n';
    }
}
