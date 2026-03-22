#include "viz/camera_ctrl.h"

#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Construction / reset
// ---------------------------------------------------------------------------
OrbitalCamera::OrbitalCamera() {
    reset();
}

void OrbitalCamera::reset() {
    distance_ = 200.0f;
    yaw_      = 0.0f;
    pitch_    = 0.3f;
    target_   = { 0.0f, 0.0f, 0.0f };

    camera_.up         = { 0.0f, 1.0f, 0.0f };
    camera_.fovy       = 45.0f;
    camera_.projection = CAMERA_PERSPECTIVE;

    first_frame_ = true;
}

// ---------------------------------------------------------------------------
// Setters
// ---------------------------------------------------------------------------
void OrbitalCamera::set_target(Vector3 target) {
    target_ = target;
}

void OrbitalCamera::set_distance(float distance) {
    distance_ = std::max(1.0f, distance);
}

// ---------------------------------------------------------------------------
// Per-frame update
//
// raylib 3.5 does not provide GetMouseDelta(), so we compute the delta
// manually from the current and previous mouse positions.
// ---------------------------------------------------------------------------
void OrbitalCamera::update() {
    Vector2 cur_mouse = GetMousePosition();

    // On the very first call we have no valid previous position
    if (first_frame_) {
        prev_mouse_  = cur_mouse;
        first_frame_ = false;
    }

    Vector2 delta = {
        cur_mouse.x - prev_mouse_.x,
        cur_mouse.y - prev_mouse_.y
    };

    // ------ Rotation: right mouse button + drag ----------------------------
    if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
        yaw_   -= delta.x * mouse_sensitivity_;
        pitch_ -= delta.y * mouse_sensitivity_;
    }

    // Clamp pitch to avoid flipping (leave a small margin)
    static constexpr float kMaxPitch =  1.50f;  // ~86 degrees
    static constexpr float kMinPitch = -1.50f;
    pitch_ = std::max(kMinPitch, std::min(kMaxPitch, pitch_));

    // ------ Zoom: scroll wheel ---------------------------------------------
    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f) {
        distance_ -= wheel * scroll_speed_;
        distance_ = std::max(1.0f, distance_);
    }

    // ------ Pan: middle mouse button + drag --------------------------------
    if (IsMouseButtonDown(MOUSE_MIDDLE_BUTTON)) {
        // Build camera-local right and up vectors from yaw/pitch
        float cos_p = std::cos(pitch_);
        float sin_p = std::sin(pitch_);
        float cos_y = std::cos(yaw_);
        float sin_y = std::sin(yaw_);

        // Forward direction (from camera toward target)
        Vector3 forward = {
            cos_p * sin_y,
            sin_p,
            cos_p * cos_y
        };

        // World-up
        Vector3 world_up = { 0.0f, 1.0f, 0.0f };

        // Right = forward x world_up
        Vector3 right = {
            forward.z * world_up.y - forward.y * world_up.z,
            forward.x * world_up.z - forward.z * world_up.x,
            forward.y * world_up.x - forward.x * world_up.y
        };

        // Normalize right
        float rlen = std::sqrt(right.x * right.x +
                               right.y * right.y +
                               right.z * right.z);
        if (rlen > 1e-6f) {
            right.x /= rlen;
            right.y /= rlen;
            right.z /= rlen;
        }

        // Camera-local up = right x forward
        Vector3 up = {
            right.y * forward.z - right.z * forward.y,
            right.z * forward.x - right.x * forward.z,
            right.x * forward.y - right.y * forward.x
        };

        float pan_speed = distance_ * 0.002f;
        target_.x += (-delta.x * right.x + delta.y * up.x) * pan_speed;
        target_.y += (-delta.x * right.y + delta.y * up.y) * pan_speed;
        target_.z += (-delta.x * right.z + delta.y * up.z) * pan_speed;
    }

    // Save current position for next frame's delta computation
    prev_mouse_ = cur_mouse;

    // ------ Compute camera position from spherical coordinates -------------
    float cos_p = std::cos(pitch_);
    camera_.position = {
        target_.x + distance_ * cos_p * std::sin(yaw_),
        target_.y + distance_ * std::sin(pitch_),
        target_.z + distance_ * cos_p * std::cos(yaw_)
    };

    camera_.target = target_;
    camera_.up     = { 0.0f, 1.0f, 0.0f };
}
