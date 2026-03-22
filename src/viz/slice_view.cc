#include "viz/slice_view.h"
#include "viz/colormap.h"

#include <algorithm>

// ---------------------------------------------------------------------------
// draw_slice
//
// Extracts a 2D plane from the NxNxN volume, maps every cell through the
// VIRIDIS colormap, uploads the result as a temporary texture and draws it
// at the requested screen position.
// ---------------------------------------------------------------------------
void draw_slice(const float* data, int N, int axis, int slice_pos,
                int screen_x, int screen_y, int screen_size) {
    // Clamp slice position
    slice_pos = std::max(0, std::min(N - 1, slice_pos));

    // The slice is always N x N pixels (before scaling)
    Image img = GenImageColor(N, N, BLACK);

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float value = 0.0f;

            switch (axis) {
                case 0:  // X axis fixed  ->  show YZ plane  (row=Y, col=Z)
                    // data index: z * N*N + y * N + x
                    value = data[col * N * N + row * N + slice_pos];
                    break;
                case 1:  // Y axis fixed  ->  show XZ plane  (row=Z, col=X)
                    value = data[row * N * N + slice_pos * N + col];
                    break;
                case 2:  // Z axis fixed  ->  show XY plane  (row=Y, col=X)
                default:
                    value = data[slice_pos * N * N + row * N + col];
                    break;
            }

            Color c = colormap(value, ColormapType::VIRIDIS);
            ImageDrawPixel(&img, col, row, c);
        }
    }

    Texture2D tex = LoadTextureFromImage(img);
    UnloadImage(img);

    // Draw scaled to fill the requested area
    Rectangle src  = { 0.0f, 0.0f,
                       static_cast<float>(N), static_cast<float>(N) };
    Rectangle dest = { static_cast<float>(screen_x),
                       static_cast<float>(screen_y),
                       static_cast<float>(screen_size),
                       static_cast<float>(screen_size) };

    DrawTexturePro(tex, src, dest, { 0, 0 }, 0.0f, WHITE);
    UnloadTexture(tex);
}
