#include <hailo/hailort.hpp>
#include <hailo/infer_model.hpp>
#include <hailo/vdevice.hpp>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace hailort;

namespace {
bool scheduler_command(const std::string &address, const std::string &command)
{
    const std::string prefix = "unix://";
    if (address.rfind(prefix, 0) != 0) {
        std::cerr << "scheduler address must use unix://\n";
        return false;
    }
    const auto path = address.substr(prefix.size());
    if (path.size() >= sizeof(sockaddr_un::sun_path)) {
        std::cerr << "scheduler socket path is too long\n";
        return false;
    }
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;
    sockaddr_un endpoint{};
    endpoint.sun_family = AF_UNIX;
    std::memcpy(endpoint.sun_path, path.c_str(), path.size() + 1);
    if (connect(fd, reinterpret_cast<sockaddr *>(&endpoint), sizeof(endpoint)) != 0) {
        std::cerr << "connect scheduler failed: " << std::strerror(errno) << '\n';
        close(fd);
        return false;
    }
    const std::string request = command + "\n";
    if (write(fd, request.data(), request.size()) != static_cast<ssize_t>(request.size())) {
        close(fd);
        return false;
    }
    char response[256]{};
    const auto size = read(fd, response, sizeof(response) - 1);
    close(fd);
    if (size < 2 || std::string(response, static_cast<size_t>(size)).rfind("OK", 0) != 0) {
        std::cerr << "scheduler rejected " << command << ": " << response;
        return false;
    }
    return true;
}

template<typename T>
bool require(const Expected<T> &result, const char *operation)
{
    if (result) return true;
    std::cerr << operation << " failed: " << result.status() << '\n';
    return false;
}

bool require_status(hailo_status status, const char *operation)
{
    if (status == HAILO_SUCCESS) return true;
    std::cerr << operation << " failed: " << status << '\n';
    return false;
}
} // namespace

int main(int argc, char **argv)
{
    std::string hef;
    uint64_t runs = 100;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--hef" && ++i < argc) hef = argv[i];
        else if (arg == "--runs" && ++i < argc) runs = std::stoull(argv[i]);
        else { std::cerr << "usage: shared_runner --hef MODEL.hef [--runs N]\n"; return 2; }
    }
    const char *scheduler = std::getenv("NPU_SCHEDULER_ADDRESS");
    const char *workload = std::getenv("NPU_WORKLOAD_ID");
    const char *share = std::getenv("NPU_SHARE");
    const bool shared = scheduler && workload && share;
    if (hef.empty() || runs == 0 || ((scheduler || workload || share) && !shared)) {
        std::cerr << "invalid model or partial scheduler allocation environment\n";
        return 2;
    }
    if (shared && !scheduler_command(scheduler, std::string("REGISTER ") + workload + " " + share)) return 3;

    hailo_vdevice_params_t params{};
    if (!require_status(hailo_init_vdevice_params(&params), "hailo_init_vdevice_params")) return 4;
    params.multi_process_service = shared;
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
    params.group_id = "K8S_NPU_SHARE";
    auto vdevice_result = VDevice::create(params);
    if (!require(vdevice_result, "VDevice::create")) return 5;
    auto vdevice = vdevice_result.release();
    auto model_result = vdevice->create_infer_model(hef);
    if (!require(model_result, "create_infer_model")) return 6;
    auto model = model_result.release();
    model->set_batch_size(1);
    auto configured_result = model->configure();
    if (!require(configured_result, "configure")) return 7;
    auto configured = configured_result.release();
    auto bindings_result = configured.create_bindings();
    if (!require(bindings_result, "create_bindings")) return 8;
    auto bindings = bindings_result.release();

    std::map<std::string, std::vector<uint8_t>> inputs, outputs;
    for (const auto &name : model->get_input_names()) {
        auto stream = model->input(name);
        if (!require(stream, "input")) return 9;
        auto &buffer = inputs[name];
        buffer.resize(stream->get_frame_size());
        auto binding = bindings.input(name);
        if (!require(binding, "input binding") || !require_status(binding->set_buffer(MemoryView(buffer.data(), buffer.size())), "input buffer")) return 9;
    }
    for (const auto &name : model->get_output_names()) {
        auto stream = model->output(name);
        if (!require(stream, "output")) return 10;
        auto &buffer = outputs[name];
        buffer.resize(stream->get_frame_size());
        auto binding = bindings.output(name);
        if (!require(binding, "output binding") || !require_status(binding->set_buffer(MemoryView(buffer.data(), buffer.size())), "output buffer")) return 10;
    }

    uint64_t max_latency_us = 0;
    uint64_t latency_sum_us = 0;
    std::vector<uint64_t> latencies;
    latencies.reserve(runs);
    const auto started = std::chrono::steady_clock::now();
    if (shared && !scheduler_command(scheduler, std::string("ACQUIRE ") + workload)) return 11;
    for (uint64_t i = 0; i < runs; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        if (!require_status(configured.run(bindings, std::chrono::seconds(30)), "inference")) return 12;
        if (shared) {
            const auto command = ((i + 1) < runs) ? "NEXT " : "COMPLETE ";
            if (!scheduler_command(scheduler, std::string(command) + workload)) return 13;
        }
        const auto latency = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
        max_latency_us = std::max(max_latency_us, static_cast<uint64_t>(latency));
        latency_sum_us += static_cast<uint64_t>(latency);
        latencies.push_back(static_cast<uint64_t>(latency));
    }
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
    if (shared) scheduler_command(scheduler, std::string("UNREGISTER ") + workload);
    std::sort(latencies.begin(), latencies.end());
    const auto p95_latency_us = latencies[(latencies.size() * 95 + 99) / 100 - 1];
    const auto average_latency_us = latency_sum_us / runs;
    const auto throughput_fps = (static_cast<double>(runs) * 1000000.0) / static_cast<double>(elapsed_us);
    std::cout << "NPU_E2E_OK mode=" << (shared ? "shared" : "exclusive")
              << " workload=" << (shared ? workload : "direct") << " share=" << (shared ? share : "1000")
              << " runs=" << runs << " elapsed_us=" << elapsed_us << " throughput_fps=" << throughput_fps
              << " average_latency_us=" << average_latency_us << " p95_latency_us=" << p95_latency_us
              << " max_latency_us=" << max_latency_us << '\n';
    return 0;
}
