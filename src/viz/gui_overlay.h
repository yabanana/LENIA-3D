#pragma once

#include "core/lenia.h"

static constexpr int MASS_HISTORY_LEN = 180;  // 3 seconds at 60fps

struct OverlayState {
    float mu;
    float sigma;
    int   radius;
    int   T;
    float threshold;       // marching-cubes isovalue
    bool  paused;
    bool  show_slices;
    bool  reset_requested;
    int   slice_axis;      // 0 = X (YZ plane), 1 = Y (XZ plane), 2 = Z (XY plane)
    int   slice_pos;
    int   num_kernels;     // number of kernel-growth pairs

    // Mass tracking for clarity
    double initial_mass;                   // mass at step 0 (set by renderer)
    double mass_history[MASS_HISTORY_LEN]; // ring buffer for sparkline
    int    mass_hist_idx;                  // next write position
    int    mass_hist_count;                // entries filled (up to MASS_HISTORY_LEN)
    bool   auto_paused;                    // auto-paused due to pattern death
};

// Populate an OverlayState from the current Lenia configuration.
OverlayState gui_init(const LeniaConfig& config);

// Record a mass sample into the sparkline ring buffer.
void gui_push_mass(OverlayState& state, double mass);

// Draw the overlay panel and allow the user to modify parameters.
// Returns true when any parameter has been changed by the user.
bool gui_draw(OverlayState& state, int fps, int iteration, double mass, bool is_3d);
