#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
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

static inline void set_pixel_rgb(std::vector<uint8_t>& rgb, int w, int h, int x, int y,
                                 uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    const size_t idx = (size_t(y) * size_t(w) + size_t(x)) * 3;
    if (idx + 2 >= rgb.size()) return;
    rgb[idx + 0] = r;
    rgb[idx + 1] = g;
    rgb[idx + 2] = b;
}

static void draw_rect(std::vector<uint8_t>& rgb, int w, int h,
                      int x0, int y0, int x1, int y1, int thickness)
{
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);
    x0 = std::max(0, std::min(x0, w - 1));
    x1 = std::max(0, std::min(x1, w - 1));
    y0 = std::max(0, std::min(y0, h - 1));
    y1 = std::max(0, std::min(y1, h - 1));

    const uint8_t R = 255, G = 0, B = 0;

    for(int t = 0; t < thickness; t++){
        int top = y0 + t, bot = y1 - t;
        int left = x0 +t, right = x1 -t;

        for(int x = left; x <= right; x++){
            set_pixel_rgb(rgb, w, h, x, top, R, G, B);
            set_pixel_rgb(rgb, w, h, x, bot, R, G, B);
        }
        for(int y = top; y <= bot; y++){
            set_pixel_rgb(rgb, w, h, left, y, R, G, B);
            set_pixel_rgb(rgb, w, h, right, y, R, G, B);
        }
    }
}

struct CamCtx {
    uint32_t cam_id = 0;
    std::string dev;
    GstElement* cap_pipe = nullptr;
    GstElement* appsink_elem = nullptr;
    GstAppSink* appsink = nullptr;

    std::atomic<uint64_t> seq{0};
};

static GstElement* build_capture_pipeline(const std::string& dev, const std::string& sink_name) {
    // RGB 640x640
    std::string desc =
        "v4l2src device=" + dev +
        " ! videoconvert ! videoscale "
        " ! video/x-raw,format=RGB,width=640,height=640 "
        " ! appsink name=" + sink_name + " emit-signals=false max-buffers=1 drop=true sync=false";

    GError* err = nullptr;
    GstElement* pipe = gst_parse_launch(desc.c_str(), &err);
    if (!pipe) {
        std::cerr << "[camera] capture pipeline failed for " << dev << ": " << (err ? err->message : "unknown") << "\n";
        if (err) g_error_free(err);
        return nullptr;
    }
    if (err) g_error_free(err);
    return pipe;
}

static GstElement* build_display_pipeline(GstElement** out_appsrc_elems, GstAppSrc** out_appsrcs) {
    // 2 rows x 4 cols, each tile 640x640 => 2560x1280
    // appsrc src0..src7 -> compositor comp -> autovideosink
    std::string desc;
    desc += "compositor name=comp "
            "sink_0::xpos=0    sink_0::ypos=0 "
            "sink_1::xpos=640  sink_1::ypos=0 "
            "sink_2::xpos=1280 sink_2::ypos=0 "
            "sink_3::xpos=1920 sink_3::ypos=0 "
            "sink_4::xpos=0    sink_4::ypos=640 "
            "sink_5::xpos=640  sink_5::ypos=640 "
            "sink_6::xpos=1280 sink_6::ypos=640 "
            "sink_7::xpos=1920 sink_7::ypos=640 "
            "! videoconvert ! autovideosink sync=false ";

    for (int i = 0; i < 8; ++i) {
        desc += "appsrc name=src" + std::to_string(i) + " is-live=true do-timestamp=true format=time ";
        desc += "! video/x-raw,format=RGB,width=640,height=640,framerate=30/1 ";
        desc += "! videoconvert ";
        desc += "! comp. ";
    }

    GError* err = nullptr;
    GstElement* pipe = gst_parse_launch(desc.c_str(), &err);
    if (!pipe) {
        std::cerr << "[camera] display pipeline create failed: " << (err ? err->message : "unknown") << "\n";
        if (err) g_error_free(err);
        return nullptr;
    }
    if (err) g_error_free(err);

    for (int i = 0; i < 8; ++i) {
        std::string name = "src" + std::to_string(i);
        GstElement* e = gst_bin_get_by_name(GST_BIN(pipe), name.c_str());
        if (!e) {
            std::cerr << "[camera] missing appsrc " << name << "\n";
            gst_object_unref(pipe);
            return nullptr;
        }
        out_appsrc_elems[i] = e;
        out_appsrcs[i] = GST_APP_SRC(e);
    }
    return pipe;
}

int main(int argc, char** argv) {
    gst_init(&argc, &argv);

    const uint32_t W  = comm::IMG_W;
    const uint32_t H  = comm::IMG_H;
    const uint32_t CH = comm::IMG_CH;
    const uint32_t image_bytes = W * H * CH;

    // SHM slot sizes
    const uint32_t rgb_slot_bytes = (uint32_t)comm::align64(sizeof(comm::RgbSlotHeader) + image_bytes);
    const uint32_t det_slot_bytes = (uint32_t)comm::align64(sizeof(comm::DetSlotHeader) + sizeof(comm::Detection) * comm::MAX_DETS);

    // Camera produces RGB and consumes DET
    comm::ShmRingProducer shm_rgb({comm::SHM_RGB_NAME, comm::TOTAL_SLOTS, rgb_slot_bytes});
    comm::ShmRingConsumer shm_det(comm::SHM_DET_NAME); // inference creates producer

    // UDS
    comm::UdsDgram sock(comm::SOCK_CAMERA_PATH);
    sock.set_nonblocking(true);

    // Build 8 capture pipelines
    std::vector<CamCtx> cams(comm::CAM_COUNT);
    for (uint32_t i = 0; i < comm::CAM_COUNT; ++i) {
        cams[i].cam_id = i;
        cams[i].dev = "/dev/video" + std::to_string(100 + i);

        std::string sink_name = "sink" + std::to_string(i);
        cams[i].cap_pipe = build_capture_pipeline(cams[i].dev, sink_name);
        if (!cams[i].cap_pipe) return 1;

        cams[i].appsink_elem = gst_bin_get_by_name(GST_BIN(cams[i].cap_pipe), sink_name.c_str());
        if (!cams[i].appsink_elem) {
            std::cerr << "[camera] appsink not found for cam " << i << "\n";
            return 1;
        }
        cams[i].appsink = GST_APP_SINK(cams[i].appsink_elem);
    }

    // Display pipeline (2x4 tile)
    GstElement* appsrc_elems[8]{};
    GstAppSrc*  appsrcs[8]{};
    GstElement* disp_pipe = build_display_pipeline(appsrc_elems, appsrcs);
    if (!disp_pipe) return 1;

    // Start pipelines
    for (auto& c : cams) gst_element_set_state(c.cap_pipe, GST_STATE_PLAYING);
    gst_element_set_state(disp_pipe, GST_STATE_PLAYING);

    std::cout << "[camera] started 8 cameras and tiled display.\n";

    std::atomic<bool> running{true};

    // Capture threads: one per camera
    std::vector<std::thread> cap_threads;
    cap_threads.reserve(comm::CAM_COUNT);

    for (uint32_t cam_id = 0; cam_id < comm::CAM_COUNT; ++cam_id) {
        cap_threads.emplace_back([&, cam_id]() {
            auto& ctx = cams[cam_id];
            std::vector<uint8_t> frame_rgb(image_bytes);

            while (running.load(std::memory_order_relaxed)) {
                GstSample* sample = gst_app_sink_pull_sample(ctx.appsink);
                if (!sample) {
                    std::cerr << "[camera] pull_sample null cam=" << cam_id << "\n";
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
                    std::cerr << "[camera] size mismatch cam=" << cam_id
                              << " got=" << map.size << " expected=" << image_bytes << "\n";
                    gst_buffer_unmap(buffer, &map);
                    gst_sample_unref(sample);
                    continue;
                }

                std::memcpy(frame_rgb.data(), map.data, image_bytes);

                gst_buffer_unmap(buffer, &map);
                gst_sample_unref(sample);

                const uint64_t pts_ns = now_steady_ns();
                const uint64_t seq = ctx.seq.fetch_add(1, std::memory_order_relaxed);

                // If detections exist for this exact seq, overlay them (best-effort)
                const uint32_t det_slot = comm::slot_index(cam_id, seq);
                if (shm_det.slots() >= comm::TOTAL_SLOTS) {
                    const uint64_t det_seen = shm_det.read_slot_seq(det_slot);
                    // if (det_seen == seq) {
                        const uint8_t* det_ptr = shm_det.slot_ptr(det_slot);
                        if (det_ptr) {
                            const auto* dh = reinterpret_cast<const comm::DetSlotHeader*>(det_ptr);
                            const auto* dets = reinterpret_cast<const comm::Detection*>(det_ptr + sizeof(comm::DetSlotHeader));
                            const uint32_t n = std::min<uint32_t>(dh->det_count, comm::MAX_DETS);

                            for (uint32_t i = 0; i < n; ++i) {
                                const auto& d = dets[i];
                                draw_rect(frame_rgb, (int)W, (int)H,
                                          (int)d.x0, (int)d.y0, (int)d.x1, (int)d.y1, 2);
                            }
                        }
                    // }
                }

                // Write RGB to SHM
                const uint32_t rgb_slot = comm::slot_index(cam_id, seq);
                uint8_t* dst = shm_rgb.slot_ptr(rgb_slot);
                if (!dst) continue;

                auto* hdr = reinterpret_cast<comm::RgbSlotHeader*>(dst);
                uint8_t* rgb_dst = dst + sizeof(comm::RgbSlotHeader);

                hdr->seq = seq;
                hdr->pts_ns = pts_ns;
                hdr->cam_id = cam_id;
                hdr->width = W;
                hdr->height = H;
                hdr->channels = CH;
                hdr->data_bytes = image_bytes;

                std::memcpy(rgb_dst, frame_rgb.data(), image_bytes);

                std::atomic_thread_fence(std::memory_order_release);
                shm_rgb.publish_slot_seq(rgb_slot, seq);

                // Notify inference
                comm::FrameReadyMsg m{};
                m.type = comm::MsgType::FRAME_READY;
                m.cam_id = cam_id;
                m.slot = rgb_slot;
                m.width = W;
                m.height = H;
                m.channels = CH;
                m.size_bytes = image_bytes;
                m.seq = seq;
                m.pts_ns = pts_ns;

                std::cout << "sizeof(FrameReadyMsg): " << sizeof(comm::FrameReadyMsg) << " sizeof(m): " << sizeof(m) << std::endl;
                sock.send_to(comm::SOCK_INFER_PATH, &m, sizeof(m));

                // Push to display for this cam
                GstBuffer* outbuf = gst_buffer_new_allocate(nullptr, image_bytes, nullptr);
                GstMapInfo outmap{};
                if (gst_buffer_map(outbuf, &outmap, GST_MAP_WRITE)) {
                    std::memcpy(outmap.data, frame_rgb.data(), image_bytes);
                    gst_buffer_unmap(outbuf, &outmap);
                    gst_app_src_push_buffer(appsrcs[cam_id], outbuf);
                } else {
                    gst_buffer_unref(outbuf);
                }

                // Drain det notifications sometimes (non-blocking, not required)
                for (int k = 0; k < 4; ++k) {
                    uint8_t rx[256];
                    const int n = sock.recv(rx, sizeof(rx));
                    if (n <= 0) break;
                }
            }
        });
    }

    // Join
    for (auto& t : cap_threads) t.join();

    running.store(false, std::memory_order_relaxed);

    // Cleanup
    for (auto& c : cams) gst_element_set_state(c.cap_pipe, GST_STATE_NULL);
    gst_element_set_state(disp_pipe, GST_STATE_NULL);

    for (auto& c : cams) {
        if (c.appsink_elem) gst_object_unref(c.appsink_elem);
        if (c.cap_pipe) gst_object_unref(c.cap_pipe);
    }
    for (int i = 0; i < 8; ++i) {
        if (appsrc_elems[i]) gst_object_unref(appsrc_elems[i]);
    }
    gst_object_unref(disp_pipe);

    return 0;
}
