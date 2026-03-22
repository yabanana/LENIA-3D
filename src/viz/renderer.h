#pragma once

#include "core/lenia.h"

#include <string>

struct RenderConfig {
    int         window_width  = 1280;
    int         window_height = 720;
    bool        fullscreen    = false;
    std::string pattern       = "random";
};

// Run the interactive visualization loop.
// This function takes ownership of the window lifecycle: it calls
// InitWindow / CloseWindow internally and returns only when the
// user closes the window.
void run_visualization(Lenia& lenia, const RenderConfig& render_config);
