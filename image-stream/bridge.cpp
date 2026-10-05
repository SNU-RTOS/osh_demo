// Binary RGB input -> existing camera IPC -> JSON detections output.
#include "ipc_protocol.hpp"
#include "shm_ring.hpp"
#include "uds_dgram.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <sys/mman.h>

static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    try {
        if (argc != 2 || (std::string(argv[1]) != "--mock" && std::string(argv[1]) != "--external"))
            throw std::runtime_error("usage: image_stream_bridge --mock|--external");
        if (comm::ipc_suffix().empty()) throw std::runtime_error("Set a unique OSH_TASK_ID");
        const bool mock = std::string(argv[1]) == "--mock";
        // The caller owns this unique session; unlink only its rings on exit.
        struct Cleanup {
            bool mock;
            ~Cleanup() {
                shm_unlink(comm::SHM_RGB_NAME);
                if (mock) shm_unlink(comm::SHM_DET_NAME);
            }
        } cleanup{mock};
        const uint32_t bytes = comm::IMG_W * comm::IMG_H * comm::IMG_CH;
        comm::ShmRingProducer rgb({comm::SHM_RGB_NAME, comm::TOTAL_SLOTS,
            uint32_t(comm::align64(sizeof(comm::RgbSlotHeader) + bytes))});
        comm::UdsDgram camera(comm::SOCK_CAMERA_PATH);
        camera.set_nonblocking(true);
        std::unique_ptr<comm::ShmRingProducer> mock_det;
        std::unique_ptr<comm::ShmRingConsumer> mock_rgb;
        std::unique_ptr<comm::UdsDgram> mock_socket;
        if (mock) {
            mock_det = std::make_unique<comm::ShmRingProducer>(comm::ShmRingConfig{
                comm::SHM_DET_NAME, comm::TOTAL_SLOTS,
                uint32_t(comm::align64(sizeof(comm::DetSlotHeader) + comm::MAX_DETS * sizeof(comm::Detection)))});
            mock_rgb = std::make_unique<comm::ShmRingConsumer>(comm::SHM_RGB_NAME);
            mock_socket = std::make_unique<comm::UdsDgram>(comm::SOCK_INFER_PATH);
            mock_socket->set_nonblocking(true);
        }
        comm::ShmRingConsumer det(comm::SHM_DET_NAME, 5000);
        if (det.slots() != comm::TOTAL_SLOTS || det.slot_bytes() < sizeof(comm::DetSlotHeader) + comm::MAX_DETS * sizeof(comm::Detection))
            throw std::runtime_error("Detection ring layout mismatch");
        uint64_t seqs[comm::CAM_COUNT]{};
        std::string frame(bytes, '\0');
        uint32_t cam;
        while (std::cin.read(reinterpret_cast<char*>(&cam), sizeof(cam))) {
            if (cam >= comm::CAM_COUNT || !std::cin.read(frame.data(), bytes))
                throw std::runtime_error("Invalid or truncated input frame");
            uint64_t seq = ++seqs[cam];
            uint32_t slot = comm::slot_index(cam, seq);
            uint64_t start = now_ns();
            comm::RgbSlotHeader rh{seq, start, cam, comm::IMG_W, comm::IMG_H, comm::IMG_CH, bytes};
            auto dst = rgb.slot_ptr(slot);
            std::memcpy(dst, &rh, sizeof(rh));
            std::memcpy(dst + sizeof(rh), frame.data(), bytes);
            rgb.publish_slot_seq(slot, seq);
            comm::FrameReadyMsg fm{comm::MsgType::FRAME_READY, cam, slot,
                comm::IMG_W, comm::IMG_H, comm::IMG_CH, bytes, seq, start};
            if (!camera.send_to(comm::SOCK_INFER_PATH, &fm, sizeof(fm)))
                throw std::runtime_error("Inference socket unavailable; start both programs together");
            if (mock) {
                comm::FrameReadyMsg received{};
                if (mock_socket->recv(&received, sizeof(received)) != sizeof(received) ||
                    received.seq != seq || mock_rgb->read_slot_seq(slot) != seq)
                    throw std::runtime_error("Mock IPC round trip failed");
                const auto input = mock_rgb->slot_ptr(slot);
                comm::RgbSlotHeader copied{};
                std::memcpy(&copied, input, sizeof(copied));
                if (copied.cam_id != cam || copied.seq != seq || copied.data_bytes != bytes ||
                    std::memcmp(input + sizeof(copied), frame.data(), bytes) != 0)
                    throw std::runtime_error("Mock RGB payload mismatch");
                // A deterministic moving marker, explicitly NOT a model prediction.
                comm::Detection marker{float(40 + seq % 200), 80, float(200 + seq % 200), 260, 1, -1};
                comm::DetSlotHeader dh{seq, start, start, now_ns(), now_ns(), cam, 1};
                auto out = mock_det->slot_ptr(slot);
                std::memcpy(out, &dh, sizeof(dh));
                std::memcpy(out + sizeof(dh), &marker, sizeof(marker));
                mock_det->publish_slot_seq(slot, seq);
                comm::DetsReadyMsg dm{comm::MsgType::DETS_READY, cam, slot, 1, 0, seq, start};
                mock_socket->send_to(comm::SOCK_CAMERA_PATH, &dm, sizeof(dm));
            }
            bool ready = false;
            while (now_ns() - start < 5000000000ULL) {
                comm::DetsReadyMsg dm{};
                int n = camera.recv(&dm, sizeof(dm));
                if (n == sizeof(dm) && dm.type == comm::MsgType::DETS_READY &&
                    dm.cam_id == cam && dm.slot == slot && dm.seq == seq && det.read_slot_seq(slot) == seq) {
                    ready = true; break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (!ready) throw std::runtime_error("Detection timeout after 5 seconds");
            comm::DetSlotHeader dh{};
            const auto src = det.slot_ptr(slot);
            std::memcpy(&dh, src, sizeof(dh));
            if (dh.seq != seq || dh.cam_id != cam || dh.det_count > comm::MAX_DETS)
                throw std::runtime_error("Detection identity or count mismatch");
            std::cout << "{\"sequence\":" << seq << ",\"channel\":" << cam
                << ",\"latency_ms\":" << (now_ns() - start) / 1e6 << ",\"detections\":[";
            for (uint32_t i = 0; i < dh.det_count; ++i) {
                comm::Detection d{};
                std::memcpy(&d, src + sizeof(dh) + i * sizeof(d), sizeof(d));
                if (!std::isfinite(d.x0) || !std::isfinite(d.y0) || !std::isfinite(d.x1) ||
                    !std::isfinite(d.y1) || !std::isfinite(d.score))
                    throw std::runtime_error("Non-finite detection values");
                if (i) std::cout << ',';
                std::cout << "[" << d.x0 << ',' << d.y0 << ',' << d.x1 << ',' << d.y1 << ',' << d.score << ',' << d.class_id << ']';
            }
            std::cout << "]}" << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
