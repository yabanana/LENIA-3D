#pragma once

#include <vector>
#include <raylib.h>

struct MCVertex {
    Vector3 position;
    Vector3 normal;
    float   scalar;   // interpolated value from optional color field
};

struct MCMesh {
    std::vector<MCVertex> vertices;
    // Every 3 consecutive vertices form a triangle.

    // Convert to a raylib Mesh struct suitable for DrawMesh/DrawModel.
    // If map_scalar is true, vertex colors are set via viridis mapping of
    // the scalar field (expected range [-1,+1] mapped to [0,1]).
    Mesh to_raylib_mesh(bool map_scalar = false) const;

    // Remove all vertices.
    void clear();
};

// Extract an isosurface from a 3D scalar field using marching cubes.
//
//   data        : pointer to an NxNxN float grid (index = z*N*N + y*N + x)
//   N           : grid dimension (assumed cubic)
//   threshold   : isovalue -- vertices above this are considered "inside"
//   step        : subsampling factor (step=2 halves resolution per axis)
//   color_field : optional second NxNxN field interpolated per vertex
//                 (e.g., growth G(U) in [-1,+1]). Pass nullptr to skip.
MCMesh extract_isosurface(const float* data, int N, float threshold = 0.1f,
                          int step = 1, const float* color_field = nullptr);
