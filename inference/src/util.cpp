#include "util.hpp"

namespace util {
    // in_w/in_h = network input resolution (or the resized letterboxed image resolution).
    std::vector<Detection> decode_hailo_nms_by_class_f32(
        const uint8_t *buf,
        int num_classes,
        int max_bboxes_per_class,
        float score_thresh,
        int in_w,
        int in_h
    ) {
        std::vector<Detection> dets;
        const uint8_t *p = buf;

        for (int c = 0; c < num_classes; ++c) {
            // bbox_count is float32 in BY_CLASS
            float bbox_count_f = *reinterpret_cast<const float*>(p);
            p += sizeof(float);

            int count = static_cast<int>(bbox_count_f);
            if (count < 0) count = 0;
            if (count > max_bboxes_per_class) count = max_bboxes_per_class;

            auto bboxes = reinterpret_cast<const HailoBboxF32*>(p);

            for (int i = 0; i < count; ++i) {
                const auto &b = bboxes[i];
                if (b.score < score_thresh) continue;

                // normalized -> pixel
                float x0n = clamp01(b.x_min);
                float y0n = clamp01(b.y_min);
                float x1n = clamp01(b.x_max);
                float y1n = clamp01(b.y_max);

                Detection d;
                d.x0 = x0n * in_w;
                d.y0 = y0n * in_h;
                d.x1 = x1n * in_w;
                d.y1 = y1n * in_h;
                d.score = b.score;
                d.class_id = c;
                dets.push_back(d);
            }

            // Advance by the *reserved* max entries (not count)
            p += max_bboxes_per_class * sizeof(HailoBboxF32);
        }

        return dets;
    }
} // end of namespace postproc