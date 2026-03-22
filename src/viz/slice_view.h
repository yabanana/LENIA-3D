#pragma once

#include <raylib.h>

// Render a 2D slice of a 3D scalar field to the screen.
//
//   data       : pointer to an NxNxN float grid (row-major: z*N*N + y*N + x)
//   N          : grid dimension (assumed cubic)
//   axis       : 0 = X (shows YZ plane), 1 = Y (shows XZ plane), 2 = Z (shows XY plane)
//   slice_pos  : position along the chosen axis, clamped to [0, N-1]
//   screen_x/y : top-left pixel coordinates on screen
//   screen_size: pixel width and height of the rendered slice
void draw_slice(const float* data, int N, int axis, int slice_pos,
                int screen_x, int screen_y, int screen_size);
