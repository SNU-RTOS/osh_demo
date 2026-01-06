#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "ipc_protocol.hpp"
#include "shm_ring.hpp"
#include "uds_dgram.hpp"
#include "util.hpp"

#include "hailo/hailort.hpp"

using namespace hailort;

static inline void short_sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int main(int argc, char *argv[]) {
    const char *hef_path = (argc >= 2) ? argv[1] : "yolov10s.hef";

    std::atomic<bool> running{true};

    // UDS: inference binds inference socket, receives from camera socket
    comm::UdsDgram sock(comm::SOCK_INFER_PATH);
    sock.set_nonblocking(true);

    // SHM: inference consumes RGB, produces DET
    comm::ShmRingConsumer shm_rgb(comm::SHM_RGB_NAME);

    const uint32_t slots = shm_rgb.slots();
    const uint32_t det_slot_bytes =
        (uint32_t)comm::align64(sizeof(comm::DetSlotHeader) + sizeof(comm::Detection) * comm::MAX_DETS);
    comm::ShmRingProducer shm_det({comm::SHM_DET_NAME, slots, det_slot_bytes});

    // ----------------------------
    // HailoRT init
    // ----------------------------
    hailo_vdevice_params_t params;
    auto status = hailo_init_vdevice_params(&params);
    if (HAILO_SUCCESS != status) {
        std::cerr << "[infer] hailo_init_vdevice_params failed status=" << status << "\n";
        return (int)status;
    }

    params.device_count = 1;
    auto vdevice = VDevice::create(params);
    if (!vdevice) {
        std::cerr << "[infer] VDevice::create failed status=" << vdevice.status() << "\n";
        return (int)vdevice.status();
    }

    auto hef = Hef::create(hef_path);
    if (!hef) {
        std::cerr << "[infer] Hef::create failed status=" << hef.status() << "\n";
        return (int)hef.status();
    }

    auto cfg_params = vdevice.value()->create_configure_params(hef.value());
    if (!cfg_params) {
        std::cerr << "[infer] create_configure_params failed status=" << cfg_params.status() << "\n";
        return (int)cfg_params.status();
    }

    auto ngs = vdevice.value()->configure(hef.value(), cfg_params.value());
    if (!ngs || ngs.value().empty()) {
        std::cerr << "[infer] configure failed status=" << ngs.status() << "\n";
        return (int)ngs.status();
    }
    auto network_group = ngs.value().at(0);

    auto in_params = network_group->make_input_vstream_params({}, HAILO_FORMAT_TYPE_AUTO,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    auto out_params = network_group->make_output_vstream_params({}, HAILO_FORMAT_TYPE_AUTO,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);

    if (!in_params || !out_params) {
        std::cerr << "[infer] make_*_vstream_params failed\n";
        return 1;
    }

    auto infer_vstreams_exp = InferVStreams::create(*network_group,
        in_params.value(), out_params.value());
    if (!infer_vstreams_exp) {
        std::cerr << "[infer] InferVStreams::create failed status=" << infer_vstreams_exp.status() << "\n";
        return (int)infer_vstreams_exp.status();
    }
    InferVStreams& infer_vstreams = infer_vstreams_exp.value();

    // IO buffers (single input/output)
    std::string in_name;
    std::string out_name;

    {
        auto& iv = infer_vstreams.get_input_vstreams().front().get();
        in_name = iv.name();
    }
    {
        auto& ov = infer_vstreams.get_output_vstreams().front().get();
        out_name = ov.name();
    }

    std::map<std::string, std::vector<uint8_t>> in_bufs;
    std::map<std::string, MemoryView> in_views;
    {
        auto& iv = infer_vstreams.get_input_vstreams().front().get();
        const size_t in_size = iv.get_frame_size();
        in_bufs[in_name] = std::vector<uint8_t>(in_size, 0);
        in_views.emplace(in_name, MemoryView(in_bufs[in_name].data(), in_bufs[in_name].size()));
        std::cout << "[infer] input=" << in_name << " bytes=" << in_size << "\n";
    }

    std::map<std::string, std::vector<uint8_t>> out_bufs;
    std::map<std::string, MemoryView> out_views;
    int C = 80;
    int B = 100;
    {
        auto& ov = infer_vstreams.get_output_vstreams().front().get();
        const size_t out_size = ov.get_frame_size();
        out_bufs[out_name] = std::vector<uint8_t>(out_size, 0);
        out_views.emplace(out_name, MemoryView(out_bufs[out_name].data(), out_bufs[out_name].size()));
        auto info = ov.get_info();
        if (info.shape.height > 0) C = (int)info.shape.height;
        if (info.shape.width > 0)  B = (int)info.shape.width;
        std::cout << "[infer] output=" << out_name << " bytes=" << out_size
                  << " C=" << C << " B=" << B << "\n";
    }

    // ----------------------------
    // Main loop: recv FRAME_READY, process, publish DET_READY
    // ----------------------------
    std::cout << "[infer] ready. waiting for frames.\n";

    comm::FrameReadyMsg fm{};
    while (running.load(std::memory_order_relaxed)) {
        // Receive frame notifications (non-blocking)
        uint8_t rx[256];
        const int n = sock.recv(rx, sizeof(rx));
        if (n <= 0) {
            short_sleep_ms(1);
            continue;
        }
        if ((size_t)n < sizeof(comm::FrameReadyMsg)) continue;

        std::memcpy(&fm, rx, sizeof(fm));
        if (fm.type != comm::MsgType::FRAME_READY) continue;

        if (fm.slot >= shm_rgb.slots()) {
            std::cerr << "[infer] invalid slot=" << fm.slot << "\n";
            continue;
        }

        // Verify SHM slot still matches this seq
        std::atomic_thread_fence(std::memory_order_acquire);
        const uint64_t seen = shm_rgb.read_slot_seq(fm.slot);
        if (seen != fm.seq) {
            std::cerr << "[infer] slot overwritten: expected seq=" << fm.seq
                      << " seen=" << seen << " slot=" << fm.slot << "\n";
            continue;
        }

        const uint8_t* src = shm_rgb.slot_ptr(fm.slot);
        if (!src) {
            std::cerr << "[infer] shm_rgb.slot_ptr null\n";
            continue;
        }

        const auto* rh = reinterpret_cast<const comm::RgbSlotHeader*>(src);
        const uint8_t* rgb = src + sizeof(comm::RgbSlotHeader);

        if (rh->data_bytes == 0 || rh->data_bytes > (comm::IMG_W * comm::IMG_H * comm::IMG_CH)) {
            std::cerr << "[infer] bad rh->data_bytes=" << rh->data_bytes << "\n";
            continue;
        }

        // Copy into hailo input buffer (assumes model expects RGB uint8 sized correctly)
        uint8_t* in_ptr = in_bufs[in_name].data();
        const size_t in_size = in_bufs[in_name].size();
        if (in_size != rh->data_bytes) {
            std::cerr << "[infer] input size mismatch in_size=" << in_size
                      << " rgb_bytes=" << rh->data_bytes << "\n";
            continue;
        }
        std::memcpy(in_ptr, rgb, in_size);

        // Run inference
        status = infer_vstreams.infer(in_views, out_views, 1);
        if (HAILO_SUCCESS != status) {
            std::cerr << "[infer] infer failed status=" << status << "\n";
            continue;
        }

        // Postprocess (your function)
        uint8_t* out_ptr = out_bufs[out_name].data();
        auto dets = util::decode_hailo_nms_by_class_f32(
            out_ptr, C, B, SCORE_THRESH, (int)comm::IMG_W, (int)comm::IMG_H);

        // Write dets to SHM det ring (same slot mapping)
        const uint32_t det_slot = (uint32_t)(fm.seq % shm_det.slots());
        uint8_t* det_dst = shm_det.slot_ptr(det_slot);
        if (!det_dst) {
            std::cerr << "[infer] shm_det.slot_ptr failed\n";
            continue;
        }

        auto* dh = reinterpret_cast<comm::DetSlotHeader*>(det_dst);
        auto* arr = reinterpret_cast<comm::Detection*>(det_dst + sizeof(comm::DetSlotHeader));

        const uint32_t count = (uint32_t)std::min<size_t>(dets.size(), comm::MAX_DETS);
        dh->seq = fm.seq;
        dh->pts_ns = fm.pts_ns;
        dh->det_count = count;

        for (uint32_t i = 0; i < count; ++i) {
            const auto& d = dets[i];
            arr[i].x0 = (float)d.x0;
            arr[i].y0 = (float)d.y0;
            arr[i].x1 = (float)d.x1;
            arr[i].y1 = (float)d.y1;
            arr[i].score = d.score;
            arr[i].class_id = d.class_id;
        }

        std::atomic_thread_fence(std::memory_order_release);
        shm_det.publish_slot_seq(det_slot, fm.seq);

        // Notify camera
        comm::DetsReadyMsg dm{};
        dm.type = comm::MsgType::DETS_READY;
        dm.slot = det_slot;
        dm.det_count = count;
        dm.seq = fm.seq;
        dm.pts_ns = fm.pts_ns;

        sock.send_to(comm::SOCK_CAMERA_PATH, &dm, sizeof(dm));

        std::cout << "[infer] seq=" << fm.seq
                  << " dets=" << count
                  << " slot(rgb)=" << fm.slot
                  << " slot(det)=" << det_slot << "\n";
    }

    return 0;
}
