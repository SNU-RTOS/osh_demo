#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

#include "shm_ring.hpp"
#include "ipc_protocol.hpp"
#include "uds_dgram.hpp"

static inline uint64_t now_steady_ns() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

static inline void set_pixel_rgb(std::vector<uint8_t> &img, int w, int h, int x, int y,
                                 uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    size_t idx = (static_cast<size_t>(y) * w + x) * 3;
    img[idx + 0] = r;
    img[idx + 1] = g;
    img[idx + 2] = b;
}

static void draw_rect_rgb(std::vector<uint8_t>& rgb, int w, int h,
                          int x0, int y0, int x1, int y1,
                          uint8_t r=255, uint8_t g=0, uint8_t b=0)
{
    int thickness = 2;
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);

    x0 = std::max(0, std::min(x0, w - 1));
    x1 = std::max(0, std::min(x1, w - 1));
    y0 = std::max(0, std::min(y0, h - 1));
    y1 = std::max(0, std::min(y1, h - 1));

    for (int t = 0; t < thickness; t++) {
        int top = y0 + t, bot = y1 - t;
        int left = x0 + t, right = x1 - t;

        for (int x = left; x <= right; x++) {
            set_pixel_rgb(rgb, w, h, x, top, r, g, b);
            set_pixel_rgb(rgb, w, h, x, bot, r, g, b);
        }
        for (int y = top; y <= bot; y++) {
            set_pixel_rgb(rgb, w, h, left, y, r, g, b);
            set_pixel_rgb(rgb, w, h, right, y, r, g, b);
        }
    }
}

int main(int argc, char** argv) {
    gst_init(&argc, &argv);

    const uint32_t W  = comm::IMG_W;
    const uint32_t H  = comm::IMG_H;
    const uint32_t CH = comm::IMG_CH;
    const uint32_t image_bytes = W * H * CH;

    // Ring sizes
    const uint32_t slots = 3;

    const uint32_t rgb_slot_bytes = (uint32_t)comm::align64(sizeof(comm::RgbSlotHeader) + image_bytes);
    const uint32_t det_slot_bytes = (uint32_t)comm::align64(sizeof(comm::DetSlotHeader) + sizeof(comm::Detection) * comm::MAX_DETS);

    // SHM: camera produces RGB, consumes DET
    comm::ShmRingProducer shm_rgb({comm::SHM_RGB_NAME, slots, rgb_slot_bytes});
    comm::ShmRingConsumer shm_det(comm::SHM_DET_NAME);

    // UDS: camera binds camera socket, sends to inference socket
    comm::UdsDgram sock(comm::SOCK_CAMERA_PATH);
    sock.set_nonblocking(true);

    // Capture pipeline (appsink)
    std::string dev = "/dev/video100";
    std::string capture_desc =
        "v4l2src device=" + dev +
        " ! videoconvert ! videoscale "
        " ! video/x-raw,format=RGB,width=640,height=640 "
        " ! appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false";

    GError* err = nullptr;
    GstElement* cap_pipe = gst_parse_launch(capture_desc.c_str(), &err);
    if (!cap_pipe) {
        std::cerr << "[camera] capture pipeline create failed: " << (err ? err->message : "unknown") << "\n";
        if (err) g_error_free(err);
        return 1;
    }
    if (err) { g_error_free(err); err = nullptr; }

    GstElement* appsink_elem = gst_bin_get_by_name(GST_BIN(cap_pipe), "sink");
    if (!appsink_elem) {
        std::cerr << "[camera] appsink not found\n";
        gst_object_unref(cap_pipe);
        return 1;
    }
    GstAppSink* appsink = GST_APP_SINK(appsink_elem);

    // Display pipeline (appsrc)
    std::string disp_desc =
        "appsrc name=src is-live=true do-timestamp=true format=time "
        "! video/x-raw,format=RGB,width=640,height=640,framerate=30/1 "
        "! videoconvert ! autovideosink sync=false";

    GstElement* disp_pipe = gst_parse_launch(disp_desc.c_str(), &err);
    if (!disp_pipe) {
        std::cerr << "[camera] display pipeline create failed: " << (err ? err->message : "unknown") << "\n";
        if (err) g_error_free(err);
        gst_object_unref(appsink_elem);
        gst_object_unref(cap_pipe);
        return 1;
    }
    if (err) { g_error_free(err); err = nullptr; }

    GstElement* appsrc_elem = gst_bin_get_by_name(GST_BIN(disp_pipe), "src");
    if (!appsrc_elem) {
        std::cerr << "[camera] appsrc not found\n";
        gst_object_unref(disp_pipe);
        gst_object_unref(appsink_elem);
        gst_object_unref(cap_pipe);
        return 1;
    }
    GstAppSrc* appsrc = GST_APP_SRC(appsrc_elem);

    gst_element_set_state(cap_pipe, GST_STATE_PLAYING);
    gst_element_set_state(disp_pipe, GST_STATE_PLAYING);

    std::cout << "[camera] running. capture=" << dev << "\n";

    std::vector<uint8_t> frame_rgb(image_bytes);
    uint64_t seq = 0;

    while (true) {
        // Pull one frame (blocking)
        GstSample* sample = gst_app_sink_pull_sample(appsink);
        if (!sample) {
            std::cerr << "[camera] gst_app_sink_pull_sample returned null\n";
            break;
        }

        GstBuffer* buffer = gst_sample_get_buffer(sample);
        if (!buffer) {
            gst_sample_unref(sample);
            continue;
        }

        GstMapInfo map{};
        if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            continue;
        }

        if (map.size != image_bytes) {
            std::cerr << "[camera] unexpected frame size=" << map.size << " expected=" << image_bytes << "\n";
            gst_buffer_unmap(buffer, &map);
            gst_sample_unref(sample);
            continue;
        }

        std::memcpy(frame_rgb.data(), map.data, image_bytes);

        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);

        const uint64_t pts_ns = now_steady_ns();

        // Check if we already have detections for this seq (usually seq-1 or earlier)
        // We use det slot = seq%slots to keep mapping consistent.
        // If inference is behind, you may see no dets. That is fine.
        {
            const uint32_t det_slot = (uint32_t)(seq % shm_det.slots());
            const uint64_t seen = shm_det.read_slot_seq(det_slot);
            // if (seen == seq) {
                const uint8_t* det_ptr = shm_det.slot_ptr(det_slot);
                if (det_ptr) {
                    const auto* dh = reinterpret_cast<const comm::DetSlotHeader*>(det_ptr);
                    const auto* dets = reinterpret_cast<const comm::Detection*>(det_ptr + sizeof(comm::DetSlotHeader));
                    const uint32_t n = std::min(dh->det_count, comm::MAX_DETS);

                    for (uint32_t i = 0; i < n; ++i) {
                        const auto& d = dets[i];
                        draw_rect_rgb(frame_rgb, (int)W, (int)H,
                                      (int)d.x0, (int)d.y0, (int)d.x1, (int)d.y1);
                    }
                }
            // }
        }

        // Write RGB to SHM
        const uint32_t slot = (uint32_t)(seq % shm_rgb.slots());
        uint8_t* dst = shm_rgb.slot_ptr(slot);
        if (!dst) {
            std::cerr << "[camera] shm_rgb.slot_ptr failed\n";
            continue;
        }

        auto* hdr = reinterpret_cast<comm::RgbSlotHeader*>(dst);
        uint8_t* rgb_dst = dst + sizeof(comm::RgbSlotHeader);

        hdr->seq = seq;
        hdr->pts_ns = pts_ns;
        hdr->width = W;
        hdr->height = H;
        hdr->channels = CH;
        hdr->data_bytes = image_bytes;
        std::memcpy(rgb_dst, frame_rgb.data(), image_bytes);

        std::atomic_thread_fence(std::memory_order_release);
        shm_rgb.publish_slot_seq(slot, seq);

        // Notify inference via UDS
        comm::FrameReadyMsg m{};
        m.type = comm::MsgType::FRAME_READY;
        m.width = W;
        m.height = H;
        m.channels = CH;
        m.size_bytes = image_bytes;
        m.slot = slot;
        m.seq = seq;
        m.pts_ns = pts_ns;

        if (!sock.send_to(comm::SOCK_INFER_PATH, &m, sizeof(m))) {
            // If inference is not running yet, this will fail. Keep going.
        }

        // Display this frame (with boxes drawn)
        GstBuffer* outbuf = gst_buffer_new_allocate(nullptr, image_bytes, nullptr);
        GstMapInfo outmap{};
        if (gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE)) {
            std::memcpy(outmap.data, frame_rgb.data(), image_bytes);
            gst_buffer_unmap(outbuf, &outmap);
            gst_app_src_push_buffer(appsrc, outbuf);
        } else {
            gst_buffer_unref(outbuf);
        }

        // Drain any det notifications (non-blocking)
        // Not required for correctness (we read from SHM), but useful for logs.
        for (;;) {
            uint8_t rx[256];
            const int n = sock.recv(rx, sizeof(rx));
            if (n <= 0) break;

            if ((size_t)n >= sizeof(comm::DetsReadyMsg)) {
                auto* dm = reinterpret_cast<comm::DetsReadyMsg*>(rx);
                if (dm->type == comm::MsgType::DETS_READY) {
                    std::cout << "[camera] dets_ready seq=" << dm->seq
                              << " count=" << dm->det_count
                              << " slot=" << dm->slot << "\n";
                }
            }
        }

        ++seq;
    }

    gst_element_set_state(cap_pipe, GST_STATE_NULL);
    gst_element_set_state(disp_pipe, GST_STATE_NULL);

    gst_object_unref(appsrc_elem);
    gst_object_unref(disp_pipe);
    gst_object_unref(appsink_elem);
    gst_object_unref(cap_pipe);

    return 0;
}
