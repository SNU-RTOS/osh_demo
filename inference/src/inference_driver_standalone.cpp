#include "hailo/hailort.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>

using namespace hailort;

// ----------------------------
// Config
// ----------------------------
static constexpr int IMG_W = 640;
static constexpr int IMG_H = 640;
static constexpr float SCORE_THRESH = 0.25f;

// ----------------------------
// Thread-safe queue
// ----------------------------
struct FrameItem {
    GstClockTime pts = GST_CLOCK_TIME_NONE;
    std::vector<uint8_t> rgb; // packed RGB, size = 640*640*3
};

class FrameQueue {
public:
    explicit FrameQueue(size_t max_items) : max_items_(max_items) {}

    void push(FrameItem &&item) {
        std::unique_lock<std::mutex> lock(m_);
        if (closed_) return;
        if (q_.size() >= max_items_) {
            q_.pop_front(); // drop oldest to keep latency low
        }
        q_.push_back(std::move(item));
        cv_.notify_one();
    }

    bool pop(FrameItem &out) {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&]{ return closed_ || !q_.empty(); });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    void close() {
        std::unique_lock<std::mutex> lock(m_);
        closed_ = true;
        cv_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<FrameItem> q_;
    size_t max_items_ = 1;
    bool closed_ = false;
};

// ----------------------------
// Display pipeline via appsrc
// ----------------------------
struct DisplayPipeline {
    GstElement *pipeline = nullptr;
    GstElement *appsrc = nullptr;
};

static DisplayPipeline create_display_pipeline(int width, int height, int fps_num = 30, int fps_den = 1)
{
    DisplayPipeline dp;
    dp.pipeline = gst_pipeline_new("display-pipeline");
    dp.appsrc = gst_element_factory_make("appsrc", "src");
    GstElement *conv = gst_element_factory_make("videoconvert", "conv");
    GstElement *sink = gst_element_factory_make("autovideosink", "sink");

    if (!dp.pipeline || !dp.appsrc || !conv || !sink) {
        std::cerr << "Failed to create display pipeline elements\n";
        return dp;
    }

    gst_bin_add_many(GST_BIN(dp.pipeline), dp.appsrc, conv, sink, nullptr);
    if (!gst_element_link_many(dp.appsrc, conv, sink, nullptr)) {
        std::cerr << "Failed to link display pipeline\n";
        return dp;
    }

    GstCaps *caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "RGB",
        "width",  G_TYPE_INT, width,
        "height", G_TYPE_INT, height,
        "framerate", GST_TYPE_FRACTION, fps_num, fps_den,
        nullptr);

    g_object_set(G_OBJECT(dp.appsrc),
        "caps", caps,
        // "is-live", TRUE,
        "format", GST_FORMAT_TIME,
        "do-timestamp", FALSE, // TRUE
        nullptr);

    gst_caps_unref(caps);
    return dp;
}

static bool push_rgb_to_appsrc(GstElement *appsrc,
                               const std::vector<uint8_t> &rgb,
                               GstClockTime pts)
{
    auto start_display = std::chrono::high_resolution_clock::now();
    GstBuffer *buf = gst_buffer_new_allocate(nullptr, rgb.size(), nullptr);
    if (!buf) return false;

    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buf);
        return false;
    }
    std::memcpy(map.data, rgb.data(), rgb.size());
    gst_buffer_unmap(buf, &map);

    if (pts != GST_CLOCK_TIME_NONE) {
        GST_BUFFER_PTS(buf) = pts;
    }

    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buf);

    auto end_display = std::chrono::high_resolution_clock::now();
    auto display_ms = std::chrono::duration<double, std::milli>(end_display - start_display).count();

    // std::cout << "Push RGB to AppSrc: " << display_ms << "ms" << std::endl;

    return (ret == GST_FLOW_OK);
}

// ----------------------------
// Simple drawing (RGB packed)
// ----------------------------
static inline void set_pixel_rgb(std::vector<uint8_t> &img, int w, int h, int x, int y,
                                 uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || y < 0 || x >= w || y >= h) return;
    size_t idx = (static_cast<size_t>(y) * w + x) * 3;
    img[idx + 0] = r;
    img[idx + 1] = g;
    img[idx + 2] = b;
}

static void draw_rect_rgb(std::vector<uint8_t> &img, int w, int h,
                          int x1, int y1, int x2, int y2,
                          uint8_t r, uint8_t g, uint8_t b, int thickness = 2)
{
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    x1 = std::max(0, std::min(x1, w - 1));
    x2 = std::max(0, std::min(x2, w - 1));
    y1 = std::max(0, std::min(y1, h - 1));
    y2 = std::max(0, std::min(y2, h - 1));

    for (int t = 0; t < thickness; t++) {
        int top = y1 + t, bot = y2 - t;
        int left = x1 + t, right = x2 - t;

        for (int x = left; x <= right; x++) {
            set_pixel_rgb(img, w, h, x, top, r, g, b);
            set_pixel_rgb(img, w, h, x, bot, r, g, b);
        }
        for (int y = top; y <= bot; y++) {
            set_pixel_rgb(img, w, h, left, y, r, g, b);
            set_pixel_rgb(img, w, h, right, y, r, g, b);
        }
    }
}

// ----------------------------
// Postprocess decode
// ----------------------------
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

// not in use
enum class BoxLayout { XYXY, YXYX, CXCYWH };

// not in use
static inline bool valid_xyxy(float x1, float y1, float x2, float y2)
{
    return (x2 > x1) && (y2 > y1) &&
           (x1 >= -0.5f && y1 >= -0.5f && x2 <= 1.5f && y2 <= 1.5f);
}

// not in use
static BoxLayout choose_layout(const float *out, int C, int B, int F)
{
    auto score_layout = [&](BoxLayout l) {
        int ok = 0;
        int checked = 0;
        for (int cls = 0; cls < std::min(C, 10); cls++) {
            for (int b = 0; b < std::min(B, 20); b++) {
                const int base = (cls * B + b) * F;
                float a0 = out[base + 0], a1 = out[base + 1], a2 = out[base + 2], a3 = out[base + 3];

                float x1, y1, x2, y2;
                if (l == BoxLayout::XYXY) { x1=a0; y1=a1; x2=a2; y2=a3; }
                else if (l == BoxLayout::YXYX) { x1=a1; y1=a0; x2=a3; y2=a2; }
                else {
                    float cx=a0, cy=a1, w=a2, h=a3;
                    x1 = cx - w/2.f; y1 = cy - h/2.f;
                    x2 = cx + w/2.f; y2 = cy + h/2.f;
                }

                checked++;
                if (valid_xyxy(x1,y1,x2,y2)) ok++;
            }
        }
        return (checked > 0) ? (float)ok / (float)checked : 0.f;
    };

    float s_xyxy = score_layout(BoxLayout::XYXY);
    float s_yxyx = score_layout(BoxLayout::YXYX);
    float s_cxcy = score_layout(BoxLayout::CXCYWH);

    if (s_cxcy >= s_xyxy && s_cxcy >= s_yxyx) return BoxLayout::CXCYWH;
    if (s_yxyx >= s_xyxy) return BoxLayout::YXYX;
    return BoxLayout::XYXY;
}

// ----------------------------
// GStreamer capture callback
// ----------------------------
struct AppContext {
    FrameQueue *queue = nullptr;
    std::atomic<bool> *running = nullptr;
};

static GstFlowReturn on_new_sample(GstAppSink *appsink, gpointer user_data)
{
    auto sample_start = std::chrono::high_resolution_clock::now();
    auto *ctx = static_cast<AppContext*>(user_data);
    if (!ctx || !ctx->running->load(std::memory_order_relaxed)) {
        return GST_FLOW_EOS;
    }

    GstSample *sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_ERROR;

    GstCaps *caps = gst_sample_get_caps(sample);
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    if (!caps || !buffer) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    GstStructure *st = gst_caps_get_structure(caps, 0);
    int width = 0, height = 0;
    gst_structure_get_int(st, "width", &width);
    gst_structure_get_int(st, "height", &height);

    if (width != IMG_W || height != IMG_H) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    const size_t expected = (size_t)IMG_W * (size_t)IMG_H * 3;
    if (map.size < expected) {
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    FrameItem item;
    item.pts = GST_BUFFER_PTS(buffer);
    item.rgb.resize(expected);
    std::memcpy(item.rgb.data(), map.data, expected);

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    ctx->queue->push(std::move(item));

    auto sample_end = std::chrono::high_resolution_clock::now();
    auto sample_ms = std::chrono::duration<double, std::milli>(sample_end - sample_start).count();
    // std::cout << "Sample time: " << sample_ms << "ms" << std::endl;

    return GST_FLOW_OK;
}

// ----------------------------
// Bus watch to quit main loop on error/EOS
// ----------------------------
struct LoopContext {
    GMainLoop *loop = nullptr;
    std::atomic<bool> *running = nullptr;
    FrameQueue *queue = nullptr;
};

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    (void)bus;
    auto *ctx = static_cast<LoopContext*>(data);

    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        std::cout << "EOS\n";
        if (ctx && ctx->running) ctx->running->store(false);
        if (ctx && ctx->queue) ctx->queue->close();
        if (ctx && ctx->loop) g_main_loop_quit(ctx->loop);
        break;

    case GST_MESSAGE_ERROR: {
        GError *err = nullptr;
        gchar *dbg = nullptr;
        gst_message_parse_error(msg, &err, &dbg);
        std::cerr << "GStreamer ERROR: " << (err ? err->message : "unknown") << "\n";
        if (dbg) std::cerr << "Debug: " << dbg << "\n";
        if (err) g_error_free(err);
        if (dbg) g_free(dbg);

        if (ctx && ctx->running) ctx->running->store(false);
        if (ctx && ctx->queue) ctx->queue->close();
        if (ctx && ctx->loop) g_main_loop_quit(ctx->loop);
        break;
    }

    default:
        break;
    }
    return TRUE;
}

// ----------------------------
// Inference worker
// ----------------------------
static void inference_worker(std::atomic<bool> &running,
                             FrameQueue &queue,
                             InferVStreams &infer_vstream,
                             const std::string &in_name,
                             std::map<std::string, std::vector<uint8_t>> &in_bufs,
                             std::map<std::string, MemoryView> &in_views,
                             const std::string &out_name,
                             std::map<std::string, std::vector<uint8_t>> &out_bufs,
                             std::map<std::string, MemoryView> &out_views,
                             GstElement *display_appsrc,
                             int C, int B)
{
    uint8_t *in_ptr = in_bufs[in_name].data();
    const size_t in_size = in_bufs[in_name].size();

    uint8_t *out_ptr = out_bufs[out_name].data();
    const size_t out_size = out_bufs[out_name].size();

    FrameItem frame;
    uint64_t n = 0;

    while (running.load(std::memory_order_relaxed)) {
        if (!queue.pop(frame)) break;

        // Sanity check and deep copy
        auto start_copy = std::chrono::high_resolution_clock::now();
        if (frame.rgb.size() != in_size) {
            std::cerr << "[worker] input size mismatch frame=" << frame.rgb.size()
                      << " expected=" << in_size << "\n";
            continue;
        }

        // Preprocess: none needed, just copy RGB uint8
        std::memcpy(in_ptr, frame.rgb.data(), in_size);
        auto end_copy = std::chrono::high_resolution_clock::now();

        // Infer
        auto start_infer = std::chrono::high_resolution_clock::now();
        auto status = infer_vstream.infer(in_views, out_views, 1);
        if (HAILO_SUCCESS != status) {
            std::cerr << "[worker] infer failed status=" << status << "\n";
            continue;
        }
        auto end_infer = std::chrono::high_resolution_clock::now();

        // Decode (float32 normalized)
        auto start_decode = std::chrono::high_resolution_clock::now();
        auto dets = decode_hailo_nms_by_class_f32(out_ptr, C, B, SCORE_THRESH, IMG_W, IMG_H);
        auto end_decode = std::chrono::high_resolution_clock::now();

        // Draw boxes
        auto start_draw = std::chrono::high_resolution_clock::now();
        // std::vector<uint8_t> annotated = frame.rgb;
        for (const auto &d : dets) {
            draw_rect_rgb(frame.rgb, IMG_W, IMG_H, d.x0, d.y0, d.x1, d.y1, 255, 0, 0, 2);
        }
        auto end_draw = std::chrono::high_resolution_clock::now();

        // Display
        auto start_display = std::chrono::high_resolution_clock::now();
        if (display_appsrc) {
            if (!push_rgb_to_appsrc(display_appsrc, frame.rgb, frame.pts)) {
                std::cerr << "[worker] failed to push to appsrc\n";
            }
        }
        auto end_display = std::chrono::high_resolution_clock::now();

        // Log every thirty frames
        n++;
        if ((n % 30) == 0) {
            auto copy_ms = std::chrono::duration<double, std::milli>(end_copy - start_copy).count();
            auto infer_ms = std::chrono::duration<double, std::milli>(end_infer - start_infer).count();
            auto decode_ms = std::chrono::duration<double, std::milli>(end_decode - start_decode).count();
            auto draw_ms = std::chrono::duration<double, std::milli>(end_draw - start_draw).count();
            auto display_ms = std::chrono::duration<double, std::milli>(end_display - start_display).count();
            std::cout << "[worker] frame=" << n
                    << " copy=" << copy_ms << "ms"
                    << " infer=" << infer_ms << "ms"
                    << " decode=" << decode_ms << "ms"
                    << " draw=" << draw_ms << "ms"
                    << " display=" << display_ms << "ms"
                    << " dets=" << dets.size()
                    << "\n";
        }
    }
}

// ----------------------------
// Main
// ----------------------------
int main(int argc, char *argv[])
{
    gst_init(&argc, &argv);

    const char *dev = (argc >= 2) ? argv[1] : "/dev/video100";
    const char *hef_path = (argc >= 3) ? argv[2] : "yolov10s.hef";
    int device_count = (argc >= 4) ? std::atoi(argv[3]) : 1;

    std::atomic<bool> running{true};
    FrameQueue queue(3);

    // ----------------------------
    // HailoRT init
    // ----------------------------
    hailo_vdevice_params_t params;
    auto status = hailo_init_vdevice_params(&params);
    if (HAILO_SUCCESS != status) {
        std::cerr << "Failed init vdevice_params, status = " << status << std::endl;
        return status;
    }

    params.device_count = device_count; // hailort::Device::scan()
    auto vdevice = VDevice::create(params);
    if (!vdevice) {
        std::cerr << "Failed to create vdevice, status=" << vdevice.status() << "\n";
        return vdevice.status();
    } else {
        std::cout << "Created VDevice with " << device_count << " Hailo-8 NPUs" << "\n";
    }

    auto hef = Hef::create(hef_path);
    if (!hef) {
        std::cerr << "Failed to create HEF from " << hef_path << ", status=" << hef.status() << "\n";
        return hef.status();
    }

    auto configure_params = vdevice.value()->create_configure_params(hef.value());
    if (!configure_params) {
        std::cerr << "Failed to create configure params, status=" << configure_params.status() << "\n";
        return configure_params.status();
    }

    auto network_groups = vdevice.value()->configure(hef.value(), configure_params.value());
    if (!network_groups) {
        std::cerr << "Failed to configure network groups, status=" << network_groups.status() << "\n";
        return network_groups.status();
    }
    if (network_groups.value().size() != 1) {
        std::cerr << "Invalid number of network groups: " << network_groups.value().size() << "\n";
        return HAILO_INTERNAL_FAILURE;
    }
    auto network_group = network_groups.value().at(0);

    auto input_params = network_group->make_input_vstream_params({}, HAILO_FORMAT_TYPE_AUTO,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!input_params) {
        std::cerr << "Failed to make input vstream params, status=" << input_params.status() << "\n";
        return input_params.status();
    }

    auto output_params = network_group->make_output_vstream_params({}, HAILO_FORMAT_TYPE_AUTO,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!output_params) {
        std::cerr << "Failed to make output vstream params, status=" << output_params.status() << "\n";
        return output_params.status();
    }

    auto infer_vstreams_exp = InferVStreams::create(*network_group,
        input_params.value(), output_params.value());
    if (!infer_vstreams_exp) {
        std::cerr << "Failed to create InferVStreams, status=" << infer_vstreams_exp.status() << "\n";
        return infer_vstreams_exp.status();
    }
    InferVStreams &infer_vstreams = infer_vstreams_exp.value();

    // Print input info
    {
        auto &iv = infer_vstreams.get_input_vstreams().front().get();
        auto info = iv.get_info();
        std::cout << "Input shape: " << info.shape.height << "x" << info.shape.width
                  << "x" << info.shape.features << "\n";
        std::cout << "Format type: " << (int)info.format.type << "\n";
        std::cout << "Format order: " << (int)info.format.order << "\n";
    }

    // Print output info and infer class/boxes-per-class from shape if possible
    int C = 80;  // default COCO
    int B = 100; // default
    std::string out_name;

    {
        auto &ov = infer_vstreams.get_output_vstreams().front().get();
        out_name = ov.name();
        auto info = ov.get_info();
        // What I have is 80 classes and 100 boxes per class, so the total frame size is 80 * (4 + 100 * 20) = 160320 bytes
        std::cout << out_name << " "
                  << " frame_size=" << ov.get_frame_size()
                  << " type=" << (int)info.format.type
                  << " order=" << (int)info.format.order 
                  << " nms_shape=" << info.nms_shape.number_of_classes
                  << " nms_boxes_per_class=" << info.nms_shape.max_bboxes_per_class
                  << " nms_max_boxes_total=" << info.nms_shape.max_bboxes_total << "\n";

        // Many Hailo NMS outputs show height=C and width=B
        if (info.shape.height > 0) C = (int)info.shape.height;
        if (info.shape.width > 0)  B = (int)info.shape.width;
    }

    // Allocate IO buffers (single input, single output)
    std::map<std::string, std::vector<uint8_t>> in_bufs;
    std::map<std::string, MemoryView> in_views;

    {
        auto &iv = infer_vstreams.get_input_vstreams().front().get();
        std::string in_name = iv.name();
        size_t in_size = iv.get_frame_size();
        in_bufs[in_name] = std::vector<uint8_t>(in_size, 0);
        in_views.emplace(in_name, MemoryView(in_bufs[in_name].data(), in_bufs[in_name].size()));
    }

    std::map<std::string, std::vector<uint8_t>> out_bufs;
    std::map<std::string, MemoryView> out_views;

    {
        auto &ov = infer_vstreams.get_output_vstreams().front().get();
        size_t out_size = ov.get_frame_size();
        out_bufs[out_name] = std::vector<uint8_t>(out_size, 0);
        out_views.emplace(out_name, MemoryView(out_bufs[out_name].data(), out_bufs[out_name].size()));
    }

    // Fetch names again
    auto &iv = infer_vstreams.get_input_vstreams().front().get();
    std::string in_name = iv.name();

    // ----------------------------
    // Display pipeline
    // ----------------------------
    auto disp = create_display_pipeline(IMG_W, IMG_H);
    if (!disp.pipeline || !disp.appsrc) {
        std::cerr << "Display pipeline creation failed\n";
        return 1;
    }
    gst_element_set_state(disp.pipeline, GST_STATE_PLAYING);

    // ----------------------------
    // Start worker thread
    // ----------------------------
    std::thread worker(inference_worker,
        std::ref(running),
        std::ref(queue),
        std::ref(infer_vstreams),
        in_name,
        std::ref(in_bufs),
        std::ref(in_views),
        out_name,
        std::ref(out_bufs),
        std::ref(out_views),
        disp.appsrc,
        C, B
    );

    // ----------------------------
    // Capture pipeline (appsink)
    // ----------------------------
    std::string capture_desc =
        std::string("v4l2src device=") + dev +
        " ! videoconvert ! videoscale "
        " ! video/x-raw,format=RGB,width=640,height=640 "
        " ! appsink name=sink emit-signals=true max-buffers=1 drop=true";

    GError *err = nullptr;
    GstElement *capture_pipeline = gst_parse_launch(capture_desc.c_str(), &err);
    if (!capture_pipeline) {
        std::cerr << "Failed to create capture pipeline: " << (err ? err->message : "unknown") << "\n";
        if (err) g_error_free(err);
        running.store(false);
        queue.close();
        worker.join();
        return 1;
    }
    if (err) { g_error_free(err); err = nullptr; }

    AppContext appctx;
    appctx.queue = &queue;
    appctx.running = &running;

    GstElement *sink = gst_bin_get_by_name(GST_BIN(capture_pipeline), "sink");
    g_signal_connect(sink, "new-sample", G_CALLBACK(on_new_sample), &appctx);
    gst_object_unref(sink);

    GstBus *bus = gst_element_get_bus(capture_pipeline);
    GMainLoop *loop = g_main_loop_new(nullptr, FALSE);

    LoopContext lctx;
    lctx.loop = loop;
    lctx.running = &running;
    lctx.queue = &queue;

    gst_bus_add_watch(bus, bus_call, &lctx);
    gst_object_unref(bus);

    gst_element_set_state(capture_pipeline, GST_STATE_PLAYING);
    std::cout << "Capturing from " << dev << " and running inference. Ctrl+C to stop.\n";

    g_main_loop_run(loop);

    // ----------------------------
    // Shutdown
    // ----------------------------
    running.store(false);
    queue.close();

    g_main_loop_unref(loop);

    gst_element_set_state(capture_pipeline, GST_STATE_NULL);
    gst_object_unref(capture_pipeline);

    gst_element_set_state(disp.pipeline, GST_STATE_NULL);
    gst_object_unref(disp.pipeline);

    worker.join();
    return 0;
}
