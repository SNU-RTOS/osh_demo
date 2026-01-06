// util.hpp
#include <vector>
#include <cstdint>

#pragma once

// ----------------------------
// Config
// ----------------------------
static constexpr int IMG_W = 640;
static constexpr int IMG_H = 640;
static constexpr float SCORE_THRESH = 0.25f;

namespace util{
    #pragma pack(push, 1)
    struct HailoBboxF32 {
        float y_min;
        float x_min;
        float y_max;
        float x_max;
        float score;
    };
    #pragma pack(pop)

    static inline float clamp01(float v) {
        if (v < 0.0f) return 0.0f;
        if (v > 1.0f) return 1.0f;
        return v;
    }

    struct Detection {
        float x0, y0, x1, y1; // pixel coords
        float score;
        int class_id;
    };

    std::vector<Detection> decode_hailo_nms_by_class_f32(const uint8_t *buf, int num_classes, int max_bboxes_per_class, float score_thresh, int in_w, int in_h);
} // end of namespace util