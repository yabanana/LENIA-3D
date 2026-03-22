#pragma once

#include <raylib.h>

class OrbitalCamera {
public:
    OrbitalCamera();

    // Call once per frame.  Reads mouse / keyboard input and updates the
    // internal camera accordingly.
    void update();

    Camera3D get_camera() const { return camera_; }

    void set_target(Vector3 target);
    void set_distance(float distance);
    void reset();

private:
    Camera3D camera_;
    float distance_          = 200.0f;
    float yaw_               = 0.0f;    // horizontal angle (radians)
    float pitch_             = 0.3f;    // vertical angle   (radians)
    Vector3 target_          = {0, 0, 0};
    float mouse_sensitivity_ = 0.003f;
    float scroll_speed_      = 10.0f;

    // Manual mouse delta tracking (raylib 3.5 lacks GetMouseDelta)
    Vector2 prev_mouse_      = {0, 0};
    bool    first_frame_     = true;
};
