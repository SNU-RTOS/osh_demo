#pragma once

#include <cstdint>
#include <cstddef>

namespace comm {

// ----------------------------
// System constants
// ----------------------------
static constexpr uint32_t IMG_W  = 640;
static constexpr uint32_t IMG_H  = 640;
static constexpr uint32_t IMG_CH = 3;

static constexpr uint32_t CAM_COUNT     = 8;  // /dev/video100 .. /dev/video107
static constexpr uint32_t SLOTS_PER_CAM = 3;  // triple buffering per camera
static constexpr uint32_t TOTAL_SLOTS   = CAM_COUNT * SLOTS_PER_CAM;

// POSIX shm object names
static constexpr const char* SHM_RGB_NAME = "/comm_rgb_shm";
static constexpr const char* SHM_DET_NAME = "/comm_det_shm";

// Unix domain datagram socket paths (must be visible to both processes)
static constexpr const char* SOCK_CAMERA_PATH = "/tmp/oshm_camera.sock";
static constexpr const char* SOCK_INFER_PATH  = "/tmp/oshm_infer.sock";

// ----------------------------
// Message protocol (UDS datagram)
// ----------------------------
enum class MsgType : uint32_t {
    FRAME_READY = 1,
    DETS_READY  = 2,
    PING        = 3
};

#pragma pack(push, 1)
struct FrameReadyMsg {
    MsgType  type;        // FRAME_READY
    uint32_t cam_id;      // 0..7
    uint32_t slot;        // 0..TOTAL_SLOTS-1 (cam_id*SLOTS_PER_CAM + (seq%SLOTS_PER_CAM))
    uint32_t width;       // 640
    uint32_t height;      // 640
    uint32_t channels;    // 3
    uint32_t size_bytes;  // width*height*channels
    uint64_t seq;         // per-camera sequence
    uint64_t pts_ns;      // timestamp
};

struct DetsReadyMsg {
    MsgType  type;        // DETS_READY
    uint32_t cam_id;      // 0..7
    uint32_t slot;        // 0..TOTAL_SLOTS-1
    uint32_t det_count;   // number of detections
    uint32_t _reserved;
    uint64_t seq;         // per-camera sequence
    uint64_t pts_ns;      // timestamp
};
#pragma pack(pop)

// ----------------------------
// Shared memory payload formats
// ----------------------------
// RGB slot layout: [RgbSlotHeader][rgb bytes...]

#pragma pack(push, 1)
struct RgbSlotHeader {
    uint64_t seq;        // per-camera sequence
    uint64_t pts_ns;
    uint32_t cam_id;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t data_bytes; // width*height*channels
};
#pragma pack(pop)

// Detection record
#pragma pack(push, 1)
struct Detection {
    float x0;     // pixel coords
    float y0;
    float x1;
    float y1;
    float score;
    int32_t class_id;
};
#pragma pack(pop)

static constexpr uint32_t MAX_DETS = 256;

// Det slot layout: [DetSlotHeader][Detection array...]
#pragma pack(push, 1)
struct DetSlotHeader {
    uint64_t seq;        // per-camera sequence
    uint64_t pts_ns;
    uint32_t cam_id;
    uint32_t det_count;  // <= MAX_DETS
};
#pragma pack(pop)

// Helpers
static inline uint32_t slot_index(uint32_t cam_id, uint64_t seq) {
    return cam_id * SLOTS_PER_CAM + static_cast<uint32_t>(seq % SLOTS_PER_CAM);
}

} // namespace comm
