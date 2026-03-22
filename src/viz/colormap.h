#pragma once

#include <raylib.h>

enum class ColormapType { VIRIDIS, INFERNO, MAGMA, GRAYSCALE };

// Map a scalar value in [0,1] to a Color using the specified colormap.
// Values outside [0,1] are clamped.
Color colormap(float value, ColormapType type = ColormapType::VIRIDIS);
