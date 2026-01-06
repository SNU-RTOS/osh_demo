#pragma once

#include <cstdint>
#include <cstddef>

namespace comm {

// ----------------------------
// System constants
// ----------------------------
static constexpr uint32_t IMG_W = 640;
static constexpr uint32_t IMG_H = 640;
static constexpr uint32_t IMG_CH = 3;

static constexpr const char* SHM_RGB_NAME  = "/comm_rgb_shm";
static constexpr const char* SHM_DET_NAME  = "/comm_det_shm";

// Two datagram sockets (each side binds one path)
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
    uint32_t width;       // 640
    uint32_t height;      // 640
    uint32_t channels;    // 3
    uint32_t size_bytes;  // width*height*channels
    uint32_t slot;        // ring slot index
    uint64_t seq;         // monotonically increasing
    uint64_t pts_ns;      // capture timestamp
};

struct DetsReadyMsg {
    MsgType  type;        // DETS_READY
    uint32_t slot;        // ring slot index
    uint32_t det_count;   // number of detections in SHM slot
    uint64_t seq;         // same seq as frame
    uint64_t pts_ns;      // propagated
};
#pragma pack(pop)

// ----------------------------
// Shared memory payload formats
// ----------------------------

// RGB slot layout: [RgbSlotHeader][rgb bytes...]
#pragma pack(push, 1)
struct RgbSlotHeader {
    uint64_t seq;
    uint64_t pts_ns;
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

static constexpr uint32_t MAX_DETS = 3;

// Det slot layout: [DetSlotHeader][Detection array...]
#pragma pack(push, 1)
struct DetSlotHeader {
    uint64_t seq;
    uint64_t pts_ns;
    uint32_t det_count;   // <= MAX_DETS
    uint32_t _reserved;
};
#pragma pack(pop)

} // namespace comm
