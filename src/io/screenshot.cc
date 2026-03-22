#include "io/screenshot.h"

#include <raylib.h>

// ---------------------------------------------------------------------------
// take_screenshot
// Thin wrapper around raylib's TakeScreenshot that accepts a std::string.
// Must be called while a raylib window is active (between InitWindow and
// CloseWindow). The screenshot captures the current framebuffer contents.
// ---------------------------------------------------------------------------
void take_screenshot(const std::string& filename) {
    TakeScreenshot(filename.c_str());
}
