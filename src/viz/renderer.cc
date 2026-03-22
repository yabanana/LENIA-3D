// renderer.cc
//
// Main visualization loop for the Lenia simulation.
// Uses raylib 4.0 for windowing and rendering, raygui for the parameter panel,
// marching cubes for 3D isosurface extraction, an orbital camera for 3D
// navigation, and a custom Blinn-Phong shader with viridis vertex coloring.

#include "viz/renderer.h"
#include "viz/colormap.h"
#include "viz/marching_cubes.h"
#include "viz/camera_ctrl.h"
#include "viz/gui_overlay.h"
#include "viz/slice_view.h"

#include <rlgl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Embedded GLSL 330 shaders — Blinn-Phong with per-vertex viridis color
// ---------------------------------------------------------------------------
static const char* phong_vs = R"glsl(
#version 330
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 mvp;

out vec3 fragPos;
out vec3 fragNormal;
out vec4 fragColor;

void main() {
    fragPos    = vertexPosition;
    fragNormal = vertexNormal;
    fragColor  = vertexColor;
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
)glsl";

static const char* phong_fs = R"glsl(
#version 330
in vec3 fragPos;
in vec3 fragNormal;
in vec4 fragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;

out vec4 finalColor;

void main() {
    vec3 norm = normalize(fragNormal);

    // Two-sided lighting
    vec3 viewDir = normalize(viewPos - fragPos);
    if (dot(norm, viewDir) < 0.0) norm = -norm;

    // Use per-vertex viridis color as base
    vec3 baseColor = fragColor.rgb;

    // Ambient
    vec3 ambient = 0.15 * baseColor;

    // Key light (diffuse)
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * 0.60 * baseColor;

    // Fill light (softer, opposite side)
    vec3 fillDir = normalize(viewPos * 2.0 - lightPos - fragPos);
    float fillDiff = max(dot(norm, fillDir), 0.0);
    vec3 fill = fillDiff * 0.20 * baseColor;

    // Specular (Blinn-Phong)
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfDir), 0.0), 48.0);
    vec3 specular = spec * 0.35 * vec3(1.0);

    // Rim light for silhouette readability
    float rim = 1.0 - max(dot(norm, viewDir), 0.0);
    rim = smoothstep(0.45, 1.0, rim) * 0.20;
    vec3 rimColor = rim * vec3(0.5, 0.7, 1.0);

    vec3 result = ambient + diffuse + fill + specular + rimColor;
    finalColor = vec4(result, 1.0);
}
)glsl";

// ---------------------------------------------------------------------------
// Internal: build a 2D RGBA texture from the grid state via colormap
// ---------------------------------------------------------------------------
static void fill_pixels(unsigned char* pixels, const float* data, int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float v = data[row * N + col];
            Color c = colormap(v, ColormapType::VIRIDIS);
            int idx = (row * N + col) * 4;
            pixels[idx + 0] = c.r;
            pixels[idx + 1] = c.g;
            pixels[idx + 2] = c.b;
            pixels[idx + 3] = 255;
        }
    }
}

// ---------------------------------------------------------------------------
// run_visualization
// ---------------------------------------------------------------------------
void run_visualization(Lenia& lenia, const RenderConfig& render_config) {
    // ---- Window setup ------------------------------------------------------
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    if (render_config.fullscreen) {
        SetConfigFlags(FLAG_FULLSCREEN_MODE);
    }

    InitWindow(render_config.window_width, render_config.window_height,
               "Lenia -- Continuous Cellular Automaton");
    SetTargetFPS(60);

    const bool is_3d = (lenia.dimension() == 3);
    const int N      = lenia.grid_size();

    // ---- GUI state ---------------------------------------------------------
    OverlayState gui = gui_init(lenia.config());

    // ---- 3D helpers --------------------------------------------------------
    OrbitalCamera cam;
    Shader phong_shader = { 0 };
    int loc_lightPos = -1;
    int loc_viewPos  = -1;

    if (is_3d) {
        float half = static_cast<float>(N) * 0.5f;
        cam.set_target({ half, half, half });
        cam.set_distance(static_cast<float>(N) * 1.8f);

        // Load Blinn-Phong shader from embedded GLSL
        phong_shader = LoadShaderFromMemory(phong_vs, phong_fs);
        loc_lightPos = GetShaderLocation(phong_shader, "lightPos");
        loc_viewPos  = GetShaderLocation(phong_shader, "viewPos");
    }

    // ---- 2D texture (created once, updated each frame) ----------------------
    unsigned char* pixels_2d = nullptr;
    Texture2D tex_2d = { 0 };
    if (!is_3d) {
        pixels_2d = static_cast<unsigned char*>(std::malloc(N * N * 4));
        std::memset(pixels_2d, 0, N * N * 4);

        Image img;
        img.data    = pixels_2d;
        img.width   = N;
        img.height  = N;
        img.mipmaps = 1;
        img.format  = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;

        tex_2d = LoadTextureFromImage(img);
    }

    // Persistent mesh for 3D isosurface
    Mesh iso_mesh    = { 0 };
    Model iso_model  = { 0 };
    bool mesh_loaded = false;
    int  mesh_capacity = 0;

    // ---- Performance instrumentation ---------------------------------------
    using Clock = std::chrono::high_resolution_clock;
    double acc_sim  = 0.0, acc_mc  = 0.0, acc_mesh = 0.0;
    double acc_draw = 0.0, acc_total = 0.0;
    int    perf_frames = 0;
    const int PERF_INTERVAL = 60;

    // Record initial mass for tracking
    gui.initial_mass = lenia.total_mass();
    gui_push_mass(gui, gui.initial_mass);

    // ---- Main loop ---------------------------------------------------------
    while (!WindowShouldClose()) {
        auto t_frame_start = Clock::now();

        // -- Handle GUI-driven parameter changes ----------------------------
        if (gui.reset_requested) {
            gui.reset_requested = false;
            lenia.init_random(lenia.config().seed);
            gui.initial_mass = lenia.total_mass();
            gui.mass_hist_idx = 0;
            gui.mass_hist_count = 0;
            gui.auto_paused = false;
            gui_push_mass(gui, gui.initial_mass);
        }

        GrowthParams gp;
        gp.mu    = gui.mu;
        gp.sigma = gui.sigma;
        lenia.set_growth_params(gp);
        lenia.set_time_step(gui.T);

        // -- Step simulation if not paused ----------------------------------
        auto t_sim_start = Clock::now();
        if (!gui.paused) {
            lenia.step();

            double cur_mass = lenia.total_mass();
            gui_push_mass(gui, cur_mass);
            if (gui.initial_mass > 0.01) {
                double ratio = cur_mass / gui.initial_mass;
                if (ratio < 0.05 && !gui.auto_paused) {
                    gui.paused = true;
                    gui.auto_paused = true;
                }
            }
        }
        auto t_sim_end = Clock::now();

        const float* data = lenia.state_data();

        // -- Begin drawing ---------------------------------------------------
        BeginDrawing();
        ClearBackground(BLACK);

        const int gui_panel_w = 250;
        const int draw_area_w = GetScreenWidth() - gui_panel_w;
        const int draw_area_h = GetScreenHeight();

        auto t_mc_start = Clock::now(), t_mc_end = t_mc_start;
        auto t_mesh_start = Clock::now(), t_mesh_end = t_mesh_start;

        if (!is_3d) {
            // ================================================================
            //  2D RENDERING
            // ================================================================
            fill_pixels(pixels_2d, data, N);
            UpdateTexture(tex_2d, pixels_2d);

            float scale = std::min(
                static_cast<float>(draw_area_w) / static_cast<float>(N),
                static_cast<float>(draw_area_h) / static_cast<float>(N));
            float draw_w = static_cast<float>(N) * scale;
            float draw_h = static_cast<float>(N) * scale;
            float off_x  = (static_cast<float>(draw_area_w) - draw_w) * 0.5f;
            float off_y  = (static_cast<float>(draw_area_h) - draw_h) * 0.5f;

            Rectangle src  = { 0, 0, static_cast<float>(N), static_cast<float>(N) };
            Rectangle dest = { off_x, off_y, draw_w, draw_h };
            DrawTexturePro(tex_2d, src, dest, { 0, 0 }, 0.0f, WHITE);

        } else {
            // ================================================================
            //  3D RENDERING
            // ================================================================

            cam.update();
            Camera3D camera = cam.get_camera();

            // Update shader uniforms: light follows camera (offset up-right)
            float lp[3] = {
                camera.position.x + 30.0f,
                camera.position.y + 60.0f,
                camera.position.z + 20.0f
            };
            float vp[3] = {
                camera.position.x,
                camera.position.y,
                camera.position.z
            };
            SetShaderValue(phong_shader, loc_lightPos, lp, SHADER_UNIFORM_VEC3);
            SetShaderValue(phong_shader, loc_viewPos,  vp, SHADER_UNIFORM_VEC3);

            // Extract isosurface with growth field for viridis coloring
            t_mc_start = Clock::now();
            int mc_step = (N >= 256) ? 2 : 1;
            const float* growth = lenia.growth_data();
            MCMesh mc = extract_isosurface(data, N, gui.threshold, mc_step, growth);
            t_mc_end = Clock::now();

            // Update GPU mesh
            t_mesh_start = Clock::now();
            int vert_count = static_cast<int>(mc.vertices.size());

            if (vert_count > 0) {
                if (vert_count > mesh_capacity) {
                    if (mesh_loaded) {
                        UnloadModel(iso_model);
                        mesh_loaded = false;
                    }
                    iso_mesh = mc.to_raylib_mesh(true);  // map_scalar = true (viridis)
                    UploadMesh(&iso_mesh, true);
                    iso_model = LoadModelFromMesh(iso_mesh);
                    iso_model.materials[0].shader = phong_shader;
                    mesh_loaded = true;
                    mesh_capacity = vert_count;
                } else {
                    iso_mesh.vertexCount  = vert_count;
                    iso_mesh.triangleCount = vert_count / 3;

                    for (int i = 0; i < vert_count; ++i) {
                        iso_mesh.vertices[i * 3 + 0] = mc.vertices[i].position.x;
                        iso_mesh.vertices[i * 3 + 1] = mc.vertices[i].position.y;
                        iso_mesh.vertices[i * 3 + 2] = mc.vertices[i].position.z;
                        iso_mesh.normals[i * 3 + 0]  = mc.vertices[i].normal.x;
                        iso_mesh.normals[i * 3 + 1]  = mc.vertices[i].normal.y;
                        iso_mesh.normals[i * 3 + 2]  = mc.vertices[i].normal.z;

                        // Update viridis vertex color from growth scalar
                        float t = (mc.vertices[i].scalar + 1.0f) * 0.5f;
                        Color c = colormap(t, ColormapType::VIRIDIS);
                        iso_mesh.colors[i * 4 + 0] = c.r;
                        iso_mesh.colors[i * 4 + 1] = c.g;
                        iso_mesh.colors[i * 4 + 2] = c.b;
                        iso_mesh.colors[i * 4 + 3] = 255;
                    }

                    UpdateMeshBuffer(iso_mesh, 0, iso_mesh.vertices,
                                     sizeof(float) * 3 * vert_count, 0);
                    UpdateMeshBuffer(iso_mesh, 2, iso_mesh.normals,
                                     sizeof(float) * 3 * vert_count, 0);
                    // Color buffer index in raylib: attribute 3
                    UpdateMeshBuffer(iso_mesh, 3, iso_mesh.colors,
                                     sizeof(unsigned char) * 4 * vert_count, 0);
                }
            } else if (mesh_loaded) {
                iso_mesh.vertexCount  = 0;
                iso_mesh.triangleCount = 0;
            }
            t_mesh_end = Clock::now();

            BeginMode3D(camera);

                // Disable backface culling so both sides of the surface are visible
                rlDisableBackfaceCulling();

                if (mesh_loaded && iso_mesh.vertexCount > 0) {
                    DrawModel(iso_model, { 0, 0, 0 }, 1.0f, WHITE);
                }

                rlEnableBackfaceCulling();

                // Bounding box
                float fn = static_cast<float>(N);
                Vector3 box_min = { 0, 0, 0 };
                Vector3 box_max = { fn, fn, fn };
                DrawBoundingBox({ box_min, box_max }, Fade(GREEN, 0.3f));

                // Axes helper
                DrawLine3D({ 0, 0, 0 }, { fn * 0.2f, 0, 0 }, RED);
                DrawLine3D({ 0, 0, 0 }, { 0, fn * 0.2f, 0 }, GREEN);
                DrawLine3D({ 0, 0, 0 }, { 0, 0, fn * 0.2f }, BLUE);

            EndMode3D();

            // Slice overlay
            if (gui.show_slices) {
                int slice_size = std::min(draw_area_w, draw_area_h) / 3;
                int slice_x = 10;
                int slice_y = draw_area_h - slice_size - 10;
                int clamped_pos = std::max(0, std::min(N - 1, gui.slice_pos));
                draw_slice(data, N, gui.slice_axis, clamped_pos,
                           slice_x, slice_y, slice_size);

                char label[64];
                const char* axis_names[] = { "X", "Y", "Z" };
                std::snprintf(label, sizeof(label), "Slice %s=%d",
                              axis_names[gui.slice_axis], clamped_pos);
                DrawText(label, slice_x, slice_y - 18, 14, YELLOW);
            }
        }

        // -- Draw GUI overlay (on top of everything) -------------------------
        gui_draw(gui, GetFPS(), lenia.iteration(), lenia.total_mass(), is_3d);

        EndDrawing();

        // -- Accumulate performance timers ----------------------------------
        auto t_frame_end = Clock::now();
        auto to_ms = [](auto d) {
            return std::chrono::duration<double, std::milli>(d).count();
        };
        acc_sim   += to_ms(t_sim_end - t_sim_start);
        acc_total += to_ms(t_frame_end - t_frame_start);
        if (is_3d) {
            acc_mc   += to_ms(t_mc_end - t_mc_start);
            acc_mesh += to_ms(t_mesh_end - t_mesh_start);
        }
        ++perf_frames;
        if (perf_frames >= PERF_INTERVAL) {
            double inv = 1.0 / perf_frames;
            std::printf("[perf] avg/frame: total=%.1f ms  sim=%.1f ms",
                        acc_total * inv, acc_sim * inv);
            if (is_3d) {
                std::printf("  mc=%.1f ms  mesh=%.1f ms",
                            acc_mc * inv, acc_mesh * inv);
            }
            std::printf("  (%.1f FPS)\n", perf_frames * 1000.0 / acc_total);
            std::fflush(stdout);
            acc_sim = acc_mc = acc_mesh = acc_draw = acc_total = 0.0;
            perf_frames = 0;
        }
    }

    // ---- Cleanup -----------------------------------------------------------
    if (pixels_2d) {
        UnloadTexture(tex_2d);
        std::free(pixels_2d);
    }
    if (is_3d) {
        UnloadShader(phong_shader);
    }
    if (mesh_loaded) {
        UnloadModel(iso_model);
    }

    CloseWindow();
}
