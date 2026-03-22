// gui_overlay.cc
//
// raygui-based parameter panel for the Lenia visualizer.
// raygui is a single-header library; the RAYGUI_IMPLEMENTATION define must
// appear in exactly one translation unit.

#define RAYGUI_IMPLEMENTATION
#include <raygui.h>

#include "viz/gui_overlay.h"

#include <cstdio>
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------
// gui_init
// ---------------------------------------------------------------------------
OverlayState gui_init(const LeniaConfig& config) {
    OverlayState s{};
    s.mu              = config.growth.mu;
    s.sigma           = config.growth.sigma;
    s.radius          = config.kernel.radius;
    s.T               = config.T;
    s.threshold       = 0.1f;
    s.paused          = false;
    s.show_slices     = false;
    s.reset_requested = false;
    s.slice_axis      = 2;   // default Z
    s.slice_pos       = 0;
    s.num_kernels     = config.num_kernels();
    s.initial_mass    = -1.0; // sentinel: not yet recorded
    s.mass_hist_idx   = 0;
    s.mass_hist_count = 0;
    s.auto_paused     = false;
    return s;
}

// ---------------------------------------------------------------------------
// gui_push_mass
// ---------------------------------------------------------------------------
void gui_push_mass(OverlayState& state, double mass) {
    state.mass_history[state.mass_hist_idx] = mass;
    state.mass_hist_idx = (state.mass_hist_idx + 1) % MASS_HISTORY_LEN;
    if (state.mass_hist_count < MASS_HISTORY_LEN)
        ++state.mass_hist_count;
}

// ---------------------------------------------------------------------------
// gui_draw
// ---------------------------------------------------------------------------
bool gui_draw(OverlayState& state, int fps, int iteration, double mass, bool is_3d) {
    bool changed = false;

    // ---- Panel dimensions --------------------------------------------------
    const int panel_w = 250;
    const int screen_w = GetScreenWidth();
    const int screen_h = GetScreenHeight();
    const int panel_x = screen_w - panel_w;
    const int panel_y = 0;

    // Semi-transparent background
    DrawRectangle(panel_x, panel_y, panel_w, screen_h,
                  Fade(DARKGRAY, 0.85f));

    // Current vertical offset for widgets
    int y = panel_y + 10;
    const int pad = 8;
    const int lbl_h = 20;
    const int slider_h = 16;
    const int x_lbl = panel_x + 10;
    const int x_ctrl = panel_x + 10;
    const int ctrl_w = panel_w - 20;

    // ---- Info section ------------------------------------------------------
    char buf[128];

    std::snprintf(buf, sizeof(buf), "FPS: %d", fps);
    DrawText(buf, x_lbl, y, 16, LIME);
    y += lbl_h + pad;

    std::snprintf(buf, sizeof(buf), "Step: %d", iteration);
    DrawText(buf, x_lbl, y, 20, RAYWHITE);
    y += 24 + pad;

    if (state.num_kernels > 1) {
        std::snprintf(buf, sizeof(buf), "Kernels: %d", state.num_kernels);
        DrawText(buf, x_lbl, y, 16, SKYBLUE);
        y += lbl_h + pad;
    }

    // Mass with percentage
    double mass_ratio = (state.initial_mass > 0.01)
                        ? (mass / state.initial_mass)
                        : 1.0;
    Color mass_color = (mass_ratio > 0.50) ? LIME
                     : (mass_ratio > 0.10) ? YELLOW
                     : RED;

    std::snprintf(buf, sizeof(buf), "Mass: %.1f (%.0f%%)",
                  mass, mass_ratio * 100.0);
    DrawText(buf, x_lbl, y, 16, mass_color);
    y += lbl_h + pad;

    // Status badge
    const char* status_text;
    Color status_color;
    if (mass_ratio < 0.05) {
        status_text = "DEAD";
        status_color = RED;
    } else if (mass_ratio < 0.30) {
        status_text = "FADING";
        status_color = ORANGE;
    } else if (mass_ratio > 2.0) {
        status_text = "EXPLODING";
        status_color = MAGENTA;
    } else {
        status_text = "ALIVE";
        status_color = LIME;
    }
    DrawRectangle(x_lbl, y, MeasureText(status_text, 18) + 12, 22,
                  Fade(status_color, 0.25f));
    DrawText(status_text, x_lbl + 6, y + 2, 18, status_color);
    y += 26 + pad;

    // Auto-pause notification
    if (state.auto_paused) {
        DrawText("(auto-paused)", x_lbl, y, 12, Fade(RED, 0.8f));
        y += 16;
    }

    // Mass sparkline
    if (state.mass_hist_count > 1) {
        const int spark_w = ctrl_w;
        const int spark_h = 40;
        DrawRectangle(x_lbl, y, spark_w, spark_h, Fade(BLACK, 0.5f));
        DrawRectangleLines(x_lbl, y, spark_w, spark_h, Fade(GRAY, 0.5f));

        // Find min/max in history for scaling
        double hist_min = 1e30, hist_max = -1e30;
        for (int i = 0; i < state.mass_hist_count; ++i) {
            double v = state.mass_history[i];
            if (v < hist_min) hist_min = v;
            if (v > hist_max) hist_max = v;
        }
        // Include 0 in range for context
        if (hist_min > 0.0) hist_min = 0.0;
        double range = hist_max - hist_min;
        if (range < 1e-6) range = 1.0;

        // Draw reference line at initial mass
        if (state.initial_mass > 0.01) {
            float ref_y = y + spark_h - static_cast<float>(
                (state.initial_mass - hist_min) / range * spark_h);
            DrawLine(x_lbl, static_cast<int>(ref_y),
                     x_lbl + spark_w, static_cast<int>(ref_y),
                     Fade(GRAY, 0.4f));
        }

        // Draw sparkline
        int oldest = (state.mass_hist_count < MASS_HISTORY_LEN)
                     ? 0
                     : state.mass_hist_idx;
        float step_x = static_cast<float>(spark_w) / (state.mass_hist_count - 1);

        for (int i = 1; i < state.mass_hist_count; ++i) {
            int idx_prev = (oldest + i - 1) % MASS_HISTORY_LEN;
            int idx_curr = (oldest + i) % MASS_HISTORY_LEN;
            float x1 = x_lbl + step_x * (i - 1);
            float x2 = x_lbl + step_x * i;
            float y1 = y + spark_h - static_cast<float>(
                (state.mass_history[idx_prev] - hist_min) / range * spark_h);
            float y2 = y + spark_h - static_cast<float>(
                (state.mass_history[idx_curr] - hist_min) / range * spark_h);
            DrawLine(static_cast<int>(x1), static_cast<int>(y1),
                     static_cast<int>(x2), static_cast<int>(y2),
                     mass_color);
        }
        y += spark_h + pad;
    }
    y += 4;

    // Separator
    DrawLine(panel_x + 5, y, panel_x + panel_w - 5, y, GRAY);
    y += pad;

    // ---- Mu slider ---------------------------------------------------------
    std::snprintf(buf, sizeof(buf), "mu: %.4f", state.mu);
    DrawText(buf, x_lbl, y, 14, RAYWHITE);
    y += lbl_h;

    float new_mu = GuiSliderBar(
        (Rectangle){ static_cast<float>(x_ctrl), static_cast<float>(y),
                     static_cast<float>(ctrl_w), static_cast<float>(slider_h) },
        "0", "0.5", state.mu, 0.0f, 0.5f);
    if (std::fabs(new_mu - state.mu) > 1e-6f) {
        state.mu = new_mu;
        changed = true;
    }
    y += slider_h + pad;

    // ---- Sigma slider ------------------------------------------------------
    std::snprintf(buf, sizeof(buf), "sigma: %.4f", state.sigma);
    DrawText(buf, x_lbl, y, 14, RAYWHITE);
    y += lbl_h;

    float new_sigma = GuiSliderBar(
        (Rectangle){ static_cast<float>(x_ctrl), static_cast<float>(y),
                     static_cast<float>(ctrl_w), static_cast<float>(slider_h) },
        "0", "0.1", state.sigma, 0.0f, 0.1f);
    if (std::fabs(new_sigma - state.sigma) > 1e-6f) {
        state.sigma = new_sigma;
        changed = true;
    }
    y += slider_h + pad;

    // ---- Threshold slider (always shown -- harmless for 2D) ----------------
    std::snprintf(buf, sizeof(buf), "threshold: %.3f", state.threshold);
    DrawText(buf, x_lbl, y, 14, RAYWHITE);
    y += lbl_h;

    float new_thr = GuiSliderBar(
        (Rectangle){ static_cast<float>(x_ctrl), static_cast<float>(y),
                     static_cast<float>(ctrl_w), static_cast<float>(slider_h) },
        "0", "0.5", state.threshold, 0.0f, 0.5f);
    if (std::fabs(new_thr - state.threshold) > 1e-6f) {
        state.threshold = new_thr;
        changed = true;
    }
    y += slider_h + pad + 4;

    // Separator
    DrawLine(panel_x + 5, y, panel_x + panel_w - 5, y, GRAY);
    y += pad;

    // ---- Pause checkbox ----------------------------------------------------
    bool new_paused = GuiCheckBox(
        (Rectangle){ static_cast<float>(x_ctrl), static_cast<float>(y),
                     20, 20 },
        "Pause", state.paused);
    if (new_paused != state.paused) {
        state.paused = new_paused;
        changed = true;
    }
    y += 20 + pad;

    // ---- Reset button ------------------------------------------------------
    state.reset_requested = GuiButton(
        (Rectangle){ static_cast<float>(x_ctrl), static_cast<float>(y),
                     static_cast<float>(ctrl_w), 30 },
        "Reset");
    if (state.reset_requested) changed = true;
    y += 30 + pad + 4;

    // ---- 3D-only controls --------------------------------------------------
    if (is_3d) {
        DrawLine(panel_x + 5, y, panel_x + panel_w - 5, y, GRAY);
        y += pad;

        DrawText("3D Slice View", x_lbl, y, 14, YELLOW);
        y += lbl_h;

        bool new_show = GuiCheckBox(
            (Rectangle){ static_cast<float>(x_ctrl), static_cast<float>(y),
                         20, 20 },
            "Show slices", state.show_slices);
        if (new_show != state.show_slices) {
            state.show_slices = new_show;
            changed = true;
        }
        y += 20 + pad;

        if (state.show_slices) {
            // Axis selector (toggle group: X / Y / Z)
            DrawText("Axis:", x_lbl, y, 14, RAYWHITE);
            y += lbl_h;

            int new_axis = GuiToggleGroup(
                (Rectangle){ static_cast<float>(x_ctrl), static_cast<float>(y),
                             60, 24 },
                "X;Y;Z", state.slice_axis);
            if (new_axis != state.slice_axis) {
                state.slice_axis = new_axis;
                changed = true;
            }
            y += 24 + pad;

            // Slice position slider
            std::snprintf(buf, sizeof(buf), "pos: %d", state.slice_pos);
            DrawText(buf, x_lbl, y, 14, RAYWHITE);
            y += lbl_h;

            // We don't know N here, so use a reasonable max and let the
            // caller clamp. Use 256 as a safe upper bound.
            float new_pos = GuiSliderBar(
                (Rectangle){ static_cast<float>(x_ctrl), static_cast<float>(y),
                             static_cast<float>(ctrl_w), static_cast<float>(slider_h) },
                "0", "N", static_cast<float>(state.slice_pos), 0.0f, 255.0f);
            int new_pos_i = static_cast<int>(std::round(new_pos));
            if (new_pos_i != state.slice_pos) {
                state.slice_pos = new_pos_i;
                changed = true;
            }
            y += slider_h + pad;
        }
    }

    return changed;
}
