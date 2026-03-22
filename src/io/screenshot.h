#pragma once

#include <string>

// Take a screenshot using raylib's TakeScreenshot function.
// The filename should include the desired extension (e.g., ".png").
// Raylib determines the format from the extension.
void take_screenshot(const std::string& filename);
