// Hardware lifecycle probe: real inference with zero-filled inputs, no decoder.
#include "hailo/hailort.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

using namespace hailort;
static volatile std::sig_atomic_t stopped = 0;
static void stop(int) { stopped = 1; }
static std::mutex output_mutex;

static int infer(const std::string &model, const std::string &id, long frames, std::atomic<bool> &failed) {
    auto device = VDevice::create(std::vector<std::string>{id});
    if (!device) return device.status();
    auto hef = Hef::create(model);
    if (!hef) return hef.status();
    auto params = device.value()->create_configure_params(hef.value());
    if (!params) return params.status();
    auto networks = device.value()->configure(hef.value(), params.value());
    if (!networks) return networks.status();
    if (networks->size() != 1) return HAILO_INVALID_ARGUMENT;
    auto network = networks->front();
    auto inputs = network->make_input_vstream_params({}, HAILO_FORMAT_TYPE_AUTO, 1000, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    auto outputs = network->make_output_vstream_params({}, HAILO_FORMAT_TYPE_AUTO, 1000, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!inputs) return inputs.status();
    if (!outputs) return outputs.status();
    auto streams = InferVStreams::create(*network, inputs.value(), outputs.value());
    if (!streams) return streams.status();
    std::map<std::string, std::vector<uint8_t>> input_buffers, output_buffers;
    std::map<std::string, MemoryView> input_views, output_views;
    for (auto &ref : streams->get_input_vstreams()) {
        auto &stream = ref.get(); auto &buffer = input_buffers[stream.name()];
        buffer.resize(stream.get_frame_size());
        input_views.emplace(stream.name(), MemoryView(buffer.data(), buffer.size()));
    }
    for (auto &ref : streams->get_output_vstreams()) {
        auto &stream = ref.get(); auto &buffer = output_buffers[stream.name()];
        buffer.resize(stream.get_frame_size());
        output_views.emplace(stream.name(), MemoryView(buffer.data(), buffer.size()));
    }
    long done = 0;
    auto next_log = std::chrono::steady_clock::now();
    while (!stopped && !failed.load() && (frames == 0 || done < frames)) {
        auto status = streams->infer(input_views, output_views, 1);
        if (status != HAILO_SUCCESS) return status;
        ++done;
        if (std::chrono::steady_clock::now() >= next_log) {
            std::lock_guard<std::mutex> lock(output_mutex);
            std::cout << "INFERENCE_OK device=" << id << " frames=" << done << std::endl;
            next_log = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        }
    }
    return HAILO_SUCCESS;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) { std::cerr << "Usage: npu_probe MODEL.hef [frames-per-device; 0=continuous]\n"; return 1; }
    std::signal(SIGTERM, stop); std::signal(SIGINT, stop);
    const char *raw = std::getenv("NPU_COUNT"); char *end = nullptr;
    long count = raw ? std::strtol(raw, &end, 10) : 1;
    if (count < 1 || count > 8 || (raw && (*raw == '\0' || *end != '\0'))) return 1;
    long frames = argc == 3 ? std::strtol(argv[2], &end, 10) : 10;
    if (frames < 0 || (argc == 3 && (*argv[2] == '\0' || *end != '\0'))) return 1;
    auto devices = Device::scan();
    if (!devices || devices->size() != static_cast<size_t>(count)) {
        std::cerr << "DEVICE_COUNT_MISMATCH expected=" << count << " discovered=" << (devices ? devices->size() : 0) << std::endl;
        return 1;
    }
    std::atomic<bool> failed{false}; std::vector<std::thread> workers;
    for (const auto &id : devices.value()) {
        std::cout << "ASSIGNED_DEVICE " << id << std::endl;
        workers.emplace_back([&, id] {
            const int status = infer(argv[1], id, frames, failed);
            if (status != HAILO_SUCCESS) {
                failed.store(true); std::lock_guard<std::mutex> lock(output_mutex);
                std::cerr << "INFERENCE_FAILED device=" << id << " status=" << status << std::endl;
            }
        });
    }
    for (auto &worker : workers) worker.join();
    std::cout << (failed.load() ? "PROBE_FAILED" : "PROBE_COMPLETE") << std::endl;
    return failed.load() ? 1 : 0;
}
