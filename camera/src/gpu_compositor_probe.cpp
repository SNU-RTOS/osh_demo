#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int kTiles = 8;
constexpr int kTileWidth = 320;
constexpr int kTileHeight = 240;
constexpr int kColumns = 4;
constexpr int kRows = 2;
constexpr int kFps = 30;

struct Options {
    std::string sink = "glimagesink";
    int frames = 120;
};

void usage(const char *name) {
    std::cerr << "Usage: " << name << " [--sink glimagesink|fakesink] [--frames N]\n";
}

bool parse_args(int argc, char **argv, Options &options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sink" && i + 1 < argc) {
            options.sink = argv[++i];
        } else if (arg == "--frames" && i + 1 < argc) {
            options.frames = std::atoi(argv[++i]);
        } else if (arg == "--help") {
            usage(argv[0]);
            return false;
        } else {
            usage(argv[0]);
            return false;
        }
    }
    return (options.sink == "glimagesink" || options.sink == "fakesink") && options.frames > 0;
}

std::vector<uint8_t> make_frame(int tile, int frame) {
    std::vector<uint8_t> pixels(kTileWidth * kTileHeight * 3);
    const uint8_t r = static_cast<uint8_t>((tile * 31 + frame * 3) & 0xff);
    const uint8_t g = static_cast<uint8_t>((tile * 67 + frame * 5) & 0xff);
    const uint8_t b = static_cast<uint8_t>((tile * 97 + frame * 7) & 0xff);
    for (int y = 0; y < kTileHeight; ++y) {
        for (int x = 0; x < kTileWidth; ++x) {
            const size_t offset = static_cast<size_t>(y * kTileWidth + x) * 3;
            const bool stripe = ((x / 16 + y / 16 + frame / 8) % 2) == 0;
            pixels[offset + 0] = stripe ? r : static_cast<uint8_t>(r / 3);
            pixels[offset + 1] = stripe ? g : static_cast<uint8_t>(g / 3);
            pixels[offset + 2] = stripe ? b : static_cast<uint8_t>(b / 3);
        }
    }
    return pixels;
}

bool bus_has_error(GstBus *bus) {
    GstMessage *message = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (!message) return false;
    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
        GError *error = nullptr;
        gchar *debug = nullptr;
        gst_message_parse_error(message, &error, &debug);
        std::cerr << "[gpu] pipeline error: " << (error ? error->message : "unknown")
                  << (debug ? std::string(" (" + std::string(debug) + ")") : "") << "\n";
        g_clear_error(&error);
        g_free(debug);
    }
    gst_message_unref(message);
    return true;
}

GstElement *build_pipeline(const Options &options, std::vector<GstAppSrc *> &sources) {
    GstElement *pipeline = gst_pipeline_new("gpu-compositor-probe");
    GstElement *mixer = gst_element_factory_make("glvideomixer", "mix");
    GstElement *sink = gst_element_factory_make(options.sink.c_str(), "sink");
    if (!pipeline || !mixer || !sink) {
        std::cerr << "[gpu] missing required GStreamer element\n";
        if (pipeline) gst_object_unref(pipeline);
        if (mixer) gst_object_unref(mixer);
        if (sink) gst_object_unref(sink);
        return nullptr;
    }
    g_object_set(mixer, "background", 1, nullptr); // black
    if (options.sink == "glimagesink") {
        g_object_set(sink, "sync", FALSE, nullptr);
    } else {
        GstElement *download = gst_element_factory_make("gldownload", "download");
        GstElement *convert = gst_element_factory_make("videoconvert", "display-convert");
        if (!download || !convert) { std::cerr << "[gpu] missing gldownload/videoconvert\n"; return nullptr; }
        gst_bin_add_many(GST_BIN(pipeline), mixer, download, convert, sink, nullptr);
        if (!gst_element_link_many(mixer, download, convert, sink, nullptr)) {
            std::cerr << "[gpu] failed to link mixer output to CPU sink\n";
            return nullptr;
        }
    }
    if (options.sink == "glimagesink") {
        gst_bin_add_many(GST_BIN(pipeline), mixer, sink, nullptr);
        if (!gst_element_link(mixer, sink)) {
            std::cerr << "[gpu] failed to link mixer output to GL sink\n";
            return nullptr;
        }
    }

    for (int i = 0; i < kTiles; ++i) {
        GstElement *source = gst_element_factory_make("appsrc", ("src" + std::to_string(i)).c_str());
        GstElement *queue = gst_element_factory_make("queue", ("queue" + std::to_string(i)).c_str());
        GstElement *upload = gst_element_factory_make("glupload", ("upload" + std::to_string(i)).c_str());
        GstElement *convert = gst_element_factory_make("glcolorconvert", ("color" + std::to_string(i)).c_str());
        if (!source || !queue || !upload || !convert) {
            std::cerr << "[gpu] missing source upload element " << i << "\n";
            return nullptr;
        }
        GstCaps *caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGB",
            "width", G_TYPE_INT, kTileWidth, "height", G_TYPE_INT, kTileHeight,
            "framerate", GST_TYPE_FRACTION, kFps, 1, nullptr);
        g_object_set(source, "is-live", TRUE, "format", GST_FORMAT_TIME,
            "block", TRUE, "do-timestamp", FALSE, "caps", caps, nullptr);
        gst_caps_unref(caps);
        gst_bin_add_many(GST_BIN(pipeline), source, queue, upload, convert, nullptr);
        if (!gst_element_link_many(source, queue, upload, convert, nullptr)) {
            std::cerr << "[gpu] failed to link source upload chain " << i << "\n";
            return nullptr;
        }
        GstPad *src_pad = gst_element_get_static_pad(convert, "src");
        GstPad *sink_pad = gst_element_request_pad_simple(mixer, "sink_%u");
        if (!src_pad || !sink_pad || gst_pad_link(src_pad, sink_pad) != GST_PAD_LINK_OK) {
            if (src_pad) gst_object_unref(src_pad);
            if (sink_pad) gst_object_unref(sink_pad);
            std::cerr << "[gpu] failed to request/link mixer input " << i << "\n";
            return nullptr;
        }
        g_object_set(sink_pad, "xpos", (i % kColumns) * kTileWidth,
            "ypos", (i / kColumns) * kTileHeight, nullptr);
        gst_object_unref(src_pad);
        gst_object_unref(sink_pad);
        sources.push_back(GST_APP_SRC(source));
    }
    return pipeline;
}
} // namespace

int main(int argc, char **argv) {
    Options options;
    if (!parse_args(argc, argv, options)) return 2;
    gst_init(&argc, &argv);

    std::vector<GstAppSrc *> sources;
    GstElement *pipeline = build_pipeline(options, sources);
    if (!pipeline) {
        std::cerr << "[gpu] could not construct OpenGL compositor pipeline\n";
        return 1;
    }
    GstBus *bus = gst_element_get_bus(pipeline);
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "[gpu] OpenGL compositor could not enter PLAYING state\n";
        gst_object_unref(bus);
        gst_object_unref(pipeline);
        return 1;
    }

    for (int frame = 0; frame < options.frames; ++frame) {
        for (int tile = 0; tile < kTiles; ++tile) {
            auto pixels = make_frame(tile, frame);
            GstBuffer *buffer = gst_buffer_new_allocate(nullptr, pixels.size(), nullptr);
            GstMapInfo map{};
            if (!buffer || !gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
                if (buffer) gst_buffer_unref(buffer);
                std::cerr << "[gpu] buffer allocation failed\n";
                gst_element_set_state(pipeline, GST_STATE_NULL);
                gst_object_unref(bus); gst_object_unref(pipeline);
                return 1;
            }
            std::memcpy(map.data, pixels.data(), pixels.size());
            gst_buffer_unmap(buffer, &map);
            GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(frame, GST_SECOND, kFps);
            GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, kFps);
            if (gst_app_src_push_buffer(sources[tile], buffer) != GST_FLOW_OK) {
                std::cerr << "[gpu] appsrc push failed\n";
                gst_element_set_state(pipeline, GST_STATE_NULL);
                gst_object_unref(bus); gst_object_unref(pipeline);
                return 1;
            }
        }
        if (bus_has_error(bus)) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(bus); gst_object_unref(pipeline);
            return 1;
        }
    }
    for (auto source : sources) gst_app_src_end_of_stream(source);
    GstMessage *finished = gst_bus_timed_pop_filtered(bus, 10 * GST_SECOND,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    const bool failed = finished && GST_MESSAGE_TYPE(finished) == GST_MESSAGE_ERROR;
    if (finished) {
        if (failed) {
            GError *e = nullptr; gst_message_parse_error(finished, &e, nullptr);
            std::cerr << "[gpu] render failed: " << (e ? e->message : "unknown") << "\n";
            g_clear_error(&e);
        }
        gst_message_unref(finished);
    }
    gst_element_set_state(pipeline, GST_STATE_NULL);
    for (auto source : sources) gst_object_unref(source);
    gst_object_unref(bus); gst_object_unref(pipeline);
    if (failed) return 1;
    std::cout << "GPU_RENDER_OK backend=OpenGL compositor=glvideomixer frames=" << options.frames
              << " sink=" << options.sink << "\n";
    return 0;
}
