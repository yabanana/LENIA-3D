#include "viz/colormap.h"

#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

struct ColorStop {
    float t;
    unsigned char r, g, b;
};

Color lerp_stops(const ColorStop* stops, int n, float t) {
    t = std::max(0.0f, std::min(1.0f, t));

    // Find the two surrounding stops
    if (t <= stops[0].t)       return { stops[0].r, stops[0].g, stops[0].b, 255 };
    if (t >= stops[n - 1].t)   return { stops[n-1].r, stops[n-1].g, stops[n-1].b, 255 };

    int i = 0;
    while (i < n - 1 && stops[i + 1].t < t) ++i;

    float local = (t - stops[i].t) / (stops[i + 1].t - stops[i].t);
    local = std::max(0.0f, std::min(1.0f, local));

    auto mix = [](unsigned char a, unsigned char b, float f) -> unsigned char {
        return static_cast<unsigned char>(
            std::round(static_cast<float>(a) * (1.0f - f) + static_cast<float>(b) * f));
    };

    return {
        mix(stops[i].r, stops[i + 1].r, local),
        mix(stops[i].g, stops[i + 1].g, local),
        mix(stops[i].b, stops[i + 1].b, local),
        255
    };
}

// ---- Viridis approximation (dark purple -> teal -> green -> yellow) ------
static const ColorStop viridis_stops[] = {
    { 0.00f,  68,   1,  84 },
    { 0.13f,  72,  36, 117 },
    { 0.25f,  56,  88, 140 },
    { 0.38f,  39, 127, 142 },
    { 0.50f,  31, 161, 135 },
    { 0.63f,  74, 194, 109 },
    { 0.75f, 159, 218,  58 },
    { 0.88f, 223, 227,  24 },
    { 1.00f, 253, 231,  37 },
};
static constexpr int viridis_n = sizeof(viridis_stops) / sizeof(viridis_stops[0]);

// ---- Inferno (black -> dark red -> orange -> yellow -> white) ------------
static const ColorStop inferno_stops[] = {
    { 0.00f,   0,   0,   4 },
    { 0.14f,  40,  11,  84 },
    { 0.29f, 101,  21, 110 },
    { 0.43f, 159,  42,  99 },
    { 0.57f, 212,  72,  66 },
    { 0.71f, 245, 125,  21 },
    { 0.86f, 250, 193,  39 },
    { 1.00f, 252, 255, 164 },
};
static constexpr int inferno_n = sizeof(inferno_stops) / sizeof(inferno_stops[0]);

// ---- Magma (black -> dark purple -> hot pink -> orange -> white) ---------
static const ColorStop magma_stops[] = {
    { 0.00f,   0,   0,   4 },
    { 0.14f,  28,  16,  68 },
    { 0.29f,  79,  18, 123 },
    { 0.43f, 137,  34, 132 },
    { 0.57f, 193,  55, 110 },
    { 0.71f, 238, 104,  60 },
    { 0.86f, 252, 181,  78 },
    { 1.00f, 252, 253, 191 },
};
static constexpr int magma_n = sizeof(magma_stops) / sizeof(magma_stops[0]);

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
Color colormap(float value, ColormapType type) {
    value = std::max(0.0f, std::min(1.0f, value));

    switch (type) {
        case ColormapType::VIRIDIS:
            return lerp_stops(viridis_stops, viridis_n, value);

        case ColormapType::INFERNO:
            return lerp_stops(inferno_stops, inferno_n, value);

        case ColormapType::MAGMA:
            return lerp_stops(magma_stops, magma_n, value);

        case ColormapType::GRAYSCALE: {
            unsigned char v = static_cast<unsigned char>(std::round(value * 255.0f));
            return { v, v, v, 255 };
        }
    }

    // Fallback (should never be reached)
    return { 0, 0, 0, 255 };
}
