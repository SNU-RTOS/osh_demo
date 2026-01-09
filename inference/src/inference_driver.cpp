#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "ipc_protocol.hpp"
#include "shm_ring.hpp"
#include "uds_dgram.hpp"
#include "util.hpp"

#include "hailo/hailort.hpp"

static inline uint64_t now_steady_ns() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

using namespace hailort;

struct WorkItem {
    comm::FrameReadyMsg msg{};
};

struct CamQueue {
    std::mutex m;
    std::condition_variable cv;
    std::deque<WorkItem> q;
};

struct InferInstance {
    std::unique_ptr<VDevice> vdevice;
    std::shared_ptr<ConfiguredNetworkGroup> network_group;
    std::unique_ptr<InferVStreams> infer_vstreams;

    std::string in_name;
    std::string out_name;

    std::map<std::string, std::vector<uint8_t>> in_bufs;
    std::map<std::string, MemoryView> in_views;

    std::map<std::string, std::vector<uint8_t>> out_bufs;
    std::map<std::string, MemoryView> out_views;

    int C = 80;
    int B = 100;
};

static Expected<InferInstance> create_infer_instance(const std::string& hef_path, uint32_t device_count)
{
    InferInstance inst;

    hailo_vdevice_params_t params;
    auto status = hailo_init_vdevice_params(&params);
    if (HAILO_SUCCESS != status) {
        return make_unexpected(status);
    }
    params.device_count = device_count;

    auto vdevice_exp = VDevice::create(params);
    if (!vdevice_exp) return make_unexpected(vdevice_exp.status());
    inst.vdevice = std::move(vdevice_exp.value());

    auto hef_exp = Hef::create(hef_path);
    if (!hef_exp) return make_unexpected(hef_exp.status());
    auto hef = hef_exp.value();

    auto cfg_params = inst.vdevice->create_configure_params(hef);
    if (!cfg_params) return make_unexpected(cfg_params.status());

    auto ngs = inst.vdevice->configure(hef, cfg_params.value());
    if (!ngs || ngs.value().empty()) return make_unexpected(ngs.status());

    inst.network_group = ngs.value().at(0);

    auto in_params = inst.network_group->make_input_vstream_params(
        {}, HAILO_FORMAT_TYPE_AUTO,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);

    auto out_params = inst.network_group->make_output_vstream_params(
        {}, HAILO_FORMAT_TYPE_AUTO,
        HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);

    if (!in_params || !out_params) return make_unexpected(HAILO_INTERNAL_FAILURE);

    auto infer_vstreams_exp = InferVStreams::create(*inst.network_group,
        in_params.value(), out_params.value());

    if (!infer_vstreams_exp) return make_unexpected(infer_vstreams_exp.status());

    inst.infer_vstreams = std::make_unique<InferVStreams>(std::move(infer_vstreams_exp.value()));

    // Assume single input/output vstream like your current yolov10s setup
    {
        auto& iv = inst.infer_vstreams->get_input_vstreams().front().get();
        inst.in_name = iv.name();
        const size_t in_size = iv.get_frame_size();
        inst.in_bufs[inst.in_name] = std::vector<uint8_t>(in_size, 0);
        inst.in_views.emplace(inst.in_name, MemoryView(inst.in_bufs[inst.in_name].data(), inst.in_bufs[inst.in_name].size()));
        std::cout << "[infer] input=" << inst.in_name << " bytes=" << in_size << "\n";
    }

    {
        auto& ov = inst.infer_vstreams->get_output_vstreams().front().get();
        inst.out_name = ov.name();
        const size_t out_size = ov.get_frame_size();
        inst.out_bufs[inst.out_name] = std::vector<uint8_t>(out_size, 0);
        inst.out_views.emplace(inst.out_name, MemoryView(inst.out_bufs[inst.out_name].data(), inst.out_bufs[inst.out_name].size()));

        auto info = ov.get_info();
        if (info.shape.height > 0) inst.C = (int)info.shape.height;
        if (info.shape.width > 0)  inst.B = (int)info.shape.width;

        std::cout << "[infer] output=" << inst.out_name << " bytes=" << out_size
                  << " C=" << inst.C << " B=" << inst.B << "\n";
    }

    return inst;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: inference_driver <path_to_hef>\n";
        return 1;
    }
    const std::string hef_path = argv[1];

    // UDS
    comm::UdsDgram sock(comm::SOCK_INFER_PATH);
    sock.set_nonblocking(true);

    // SHM: inference consumes RGB, produces DET
    comm::ShmRingConsumer shm_rgb(comm::SHM_RGB_NAME);

    const uint32_t det_slot_bytes =
        (uint32_t)comm::align64(sizeof(comm::DetSlotHeader) + sizeof(comm::Detection) * comm::MAX_DETS);
    comm::ShmRingProducer shm_det({comm::SHM_DET_NAME, comm::TOTAL_SLOTS, det_slot_bytes});

    if (shm_rgb.slots() != comm::TOTAL_SLOTS) {
        std::cerr << "[infer] shm_rgb slots mismatch. expected=" << comm::TOTAL_SLOTS
                  << " got=" << shm_rgb.slots() << "\n";
        return 1;
    }

    // Queues
    CamQueue queues[comm::CAM_COUNT];
    std::atomic<bool> running{true};

    // Receiver thread: demux by cam_id
    std::thread rx_thread([&]() {
        while (running.load(std::memory_order_relaxed)) {
            uint8_t buf[256];
            const int n = sock.recv(buf, sizeof(buf));
            if (n <= 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if ((size_t)n < sizeof(comm::FrameReadyMsg)) continue;
            
            comm::FrameReadyMsg m{};
            std::memcpy(&m, buf, sizeof(m));
            if (m.type != comm::MsgType::FRAME_READY) continue;
            if (m.cam_id >= comm::CAM_COUNT) continue;

            {
                std::lock_guard<std::mutex> lk(queues[m.cam_id].m);
                queues[m.cam_id].q.push_back(WorkItem{m});
                // keep queue bounded to avoid runaway if camera outruns inference
                if (queues[m.cam_id].q.size() > 2) {
                    queues[m.cam_id].q.pop_front();
                }
            }
            queues[m.cam_id].cv.notify_one();
        }
    });

    // Create 8 worker inference instances
    // Using device_count=1 allocates 1 NPU for each camera input.
    // Each worker has its own VDevice / network group to avoid serialization.
    std::vector<std::thread> workers;
    workers.reserve(comm::CAM_COUNT);

    for (uint32_t cam_id = 0; cam_id < comm::CAM_COUNT; ++cam_id) {
        workers.emplace_back([&, cam_id]() {
            auto inst_exp = create_infer_instance(hef_path, 1);
            if (!inst_exp) {
                std::cerr << "[infer] failed to create infer instance cam=" << cam_id
                          << " status=" << inst_exp.status() << "\n";
                return;
            }
            auto inst = std::move(inst_exp.value());

            const size_t expected_rgb = comm::IMG_W * comm::IMG_H * comm::IMG_CH;

            while (running.load(std::memory_order_relaxed)) {
                WorkItem item{};
                {
                    std::unique_lock<std::mutex> lk(queues[cam_id].m);
                    queues[cam_id].cv.wait(lk, [&] {
                        return !queues[cam_id].q.empty() || !running.load(std::memory_order_relaxed);
                    });
                    if (!running.load(std::memory_order_relaxed)) break;
                    item = queues[cam_id].q.back();
                    queues[cam_id].q.clear(); // take latest only
                }

                const auto& fm = item.msg;

                // Validate slot and seq
                if (fm.slot >= comm::TOTAL_SLOTS) continue;

                const uint64_t seen = shm_rgb.read_slot_seq(fm.slot);
                if (seen != fm.seq) {
                    // overwritten or producer ahead, skip
                    continue;
                }

                const uint8_t* src = shm_rgb.slot_ptr(fm.slot);
                if (!src) continue;

                const auto* rh = reinterpret_cast<const comm::RgbSlotHeader*>(src);
                const uint8_t* rgb = src + sizeof(comm::RgbSlotHeader);

                if (rh->cam_id != cam_id) continue;
                if (rh->data_bytes != expected_rgb) continue;

                // Copy into Hailo input buffer
                auto& in_vec = inst.in_bufs[inst.in_name];
                if (in_vec.size() != rh->data_bytes) continue;
                std::memcpy(in_vec.data(), rgb, in_vec.size());

                // Run inference
                auto start_infer_ns = now_steady_ns();
                auto status = inst.infer_vstreams->infer(inst.in_views, inst.out_views, 1);
                if (HAILO_SUCCESS != status) {
                    std::cerr << "[infer] infer failed cam=" << cam_id << " status=" << status << "\n";
                    continue;
                }
                auto end_infer_ns = now_steady_ns();

                uint8_t* out_ptr = inst.out_bufs[inst.out_name].data();
                auto dets = util::decode_hailo_nms_by_class_f32(
                    out_ptr, inst.C, inst.B, SCORE_THRESH,
                    (int)comm::IMG_W, (int)comm::IMG_H);

                // Write dets into SHM
                const uint32_t det_slot = comm::slot_index(cam_id, fm.seq);
                uint8_t* det_dst = shm_det.slot_ptr(det_slot);
                if (!det_dst) continue;

                auto* dh = reinterpret_cast<comm::DetSlotHeader*>(det_dst);
                auto* arr = reinterpret_cast<comm::Detection*>(det_dst + sizeof(comm::DetSlotHeader));

                const uint32_t count = (uint32_t)std::min<size_t>(dets.size(), comm::MAX_DETS);

                dh->seq = fm.seq;
                dh->cap_ns = fm.cap_ns;
                dh->start_infer_ns = start_infer_ns;
                dh->end_infer_ns = end_infer_ns;
                dh->det_push_ns = now_steady_ns();
                dh->cam_id = cam_id;
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

                // Notify camera (optional; camera can also just poll shm)
                comm::DetsReadyMsg dm{};
                dm.type = comm::MsgType::DETS_READY;
                dm.cam_id = cam_id;
                dm.slot = det_slot;
                dm.det_count = count;
                dm.seq = fm.seq;
                dm.cap_ns = fm.cap_ns;

                sock.send_to(comm::SOCK_CAMERA_PATH, &dm, sizeof(dm));
            }
        });
    }

    std::cout << "[infer] running 8 workers for 8 cameras.\n";

    // Wait forever (or add signal handling if you want)
    for (auto& t : workers) t.join();

    running.store(false, std::memory_order_relaxed);
    rx_thread.join();

    return 0;
}
