#include <hailo/hailort.hpp>
#include <hailo/infer_model.hpp>
#include <hailo/vdevice.hpp>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <cmath>
#include <sys/wait.h>
#include "pacing.hpp"

using namespace hailort;

namespace {
std::atomic<bool> stop_requested{false};
std::atomic<bool> broker_failed{false};

void stop_handler(int) { stop_requested.store(true); }

bool scheduler_command(const std::string &address, const std::string &command, std::string *reply = nullptr)
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
    const std::string result(response, size > 0 ? static_cast<size_t>(size) : 0);
    if (size < 2 || result.rfind("OK", 0) != 0) {
        std::cerr << "scheduler rejected " << command << ": " << response;
        return false;
    }
    if (reply) *reply = result;
    return true;
}

class RegistrationGuard {
public:
    RegistrationGuard(const char *scheduler, const char *workload, bool active) :
        m_scheduler(scheduler ? scheduler : ""), m_workload(workload ? workload : ""), m_active(active)
    {
        if (m_active) {
            m_heartbeat = std::thread([this] {
                std::unique_lock<std::mutex> lock(m_mutex);
                while (!m_condition.wait_for(lock, std::chrono::seconds(10), [this] { return m_stop; })) {
                    lock.unlock();
                    if (!scheduler_command(m_scheduler, "HEARTBEAT " + m_workload)) {
                        broker_failed.store(true);
                        stop_requested.store(true);
                        return;
                    }
                    lock.lock();
                }
            });
        }
    }
    ~RegistrationGuard()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_condition.notify_all();
        if (m_heartbeat.joinable()) m_heartbeat.join();
        if (m_active) scheduler_command(m_scheduler, "UNREGISTER " + m_workload);
    }
private:
    std::string m_scheduler;
    std::string m_workload;
    bool m_active;
    bool m_stop = false;
    std::mutex m_mutex;
    std::condition_variable m_condition;
    std::thread m_heartbeat;
};

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
    std::signal(SIGINT, stop_handler);
    std::signal(SIGTERM, stop_handler);
    std::string hef;
    uint64_t runs = 100;
    uint64_t warmup = 0;
    std::string profile_output;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--hef" && ++i < argc) hef = argv[i];
        else if (arg == "--runs" && ++i < argc) runs = std::stoull(argv[i]);
        else if (arg == "--warmup" && ++i < argc) warmup = std::stoull(argv[i]);
        else if (arg == "--profile-json" && ++i < argc) profile_output = argv[i];
        else { std::cerr << "usage: shared_runner --hef MODEL.hef [--runs N]\n"; return 2; }
    }
    const char *scheduler = std::getenv("NPU_SCHEDULER_ADDRESS");
    const char *workload = std::getenv("NPU_WORKLOAD_ID");
    const char *share = std::getenv("NPU_SHARE");
    const bool shared = scheduler && workload && share;
    double target_fps = 0;
    if (const char *raw = std::getenv("NPU_TARGET_FPS")) {
        char *end = nullptr;
        target_fps = std::strtod(raw, &end);
        if (end == raw || *end || !std::isfinite(target_fps) || target_fps <= 0 || target_fps > 1000000) {
            std::cerr << "NPU_TARGET_FPS must be positive and <=1000000\n"; return 2;
        }
    }
    if (shared && (warmup || !profile_output.empty())) {
        std::cerr << "Profiling requires exclusive execution\n"; return 2;
    }
    if (hef.empty() || runs == 0 || ((scheduler || workload || share) && !shared)) {
        std::cerr << "invalid model or partial scheduler allocation environment\n";
        return 2;
    }
    // Verify best-effort model identity without invoking a shell.
    if (const char *expected = std::getenv("NPU_MODEL_SHA256")) {
        int fds[2];
        if (pipe(fds) != 0) return 2;
        pid_t child = fork();
        if (child == 0) {
            dup2(fds[1], STDOUT_FILENO); close(fds[0]); close(fds[1]);
            execlp("sha256sum", "sha256sum", "--", hef.c_str(), static_cast<char *>(nullptr));
            _exit(127);
        }
        close(fds[1]);
        char digest[64]; size_t got = 0;
        while (got < sizeof(digest)) { auto n=read(fds[0],digest+got,sizeof(digest)-got); if(n<=0)break;got+=n; }
        close(fds[0]); int status = 0;
        if (child < 0 || waitpid(child,&status,0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0 || got != 64 || std::string(digest,64) != expected) {
            std::cerr << "Model checksum verification failed\n"; return 2;
        }
    }
    std::string register_reply;
    std::string registration_command;
    if (shared) {
        registration_command=std::string("REGISTER ")+workload+" "+share;
        if(const char* demand=std::getenv("NPU_RESERVED_DEMAND")) {
            registration_command+=" "+std::string(demand)+" utilization-best-effort";
        }
        if (!scheduler_command(scheduler,registration_command,&register_reply)) return 3;
    }
    if (shared) std::cout << "NPU_BROKER_EPOCH " << register_reply.substr(3) << std::flush;
    RegistrationGuard registration(scheduler, workload, shared);

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
    for(uint64_t i=0;i<warmup && !stop_requested.load();++i) {
        if (!require_status(configured.run(bindings,std::chrono::seconds(30)),"warmup")) return 12;
    }
    FramePacer pacer(target_fps);
    const auto started = std::chrono::steady_clock::now();
    if (shared && target_fps == 0 && !scheduler_command(scheduler, std::string("ACQUIRE ") + workload)) return 11;
    for (uint64_t i = 0; i < runs && !stop_requested.load(); ++i) {
        if (target_fps > 0) {
            const auto deadline = pacer.deadline(std::chrono::steady_clock::now());
            while (!stop_requested.load() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_until(std::min(deadline,std::chrono::steady_clock::now()+std::chrono::milliseconds(10)));
            if (stop_requested.load()) break;
            pacer.started(std::chrono::steady_clock::now());
            if (shared && !scheduler_command(scheduler,std::string("ACQUIRE ")+workload)) return 11;
        }
        const auto begin = std::chrono::steady_clock::now();
        if (!require_status(configured.run(bindings, std::chrono::seconds(30)), "inference")) return 12;
        if (shared) {
            const auto command = (target_fps == 0 && (i + 1) < runs && !stop_requested.load()) ? "NEXT " : "COMPLETE ";
            if (!scheduler_command(scheduler, std::string(command) + workload)) return 13;
        }
        const auto latency = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
        max_latency_us = std::max(max_latency_us, static_cast<uint64_t>(latency));
        latency_sum_us += static_cast<uint64_t>(latency);
        latencies.push_back(static_cast<uint64_t>(latency));
    }
    const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
    if (broker_failed.load()) return 15;
    if (latencies.empty()) return stop_requested.load() ? 0 : 14;
    std::sort(latencies.begin(), latencies.end());
    const auto p95_latency_us = latencies[(latencies.size() * 95 + 99) / 100 - 1];
    const auto average_latency_us = latency_sum_us / latencies.size();
    if (!profile_output.empty()) {
        if (stop_requested.load() || latencies.size() != runs) return 16;
        std::ofstream file(profile_output);
        file << "{\"samples\":" << latencies.size() << ",\"warmup\":" << warmup
             << ",\"meanSeconds\":" << (static_cast<double>(latency_sum_us)/latencies.size()/1e6)
             << ",\"p95Seconds\":" << p95_latency_us/1e6 << "}\n";
        if (!file) return 16;
    }
    const auto throughput_fps = (static_cast<double>(latencies.size()) * 1000000.0) / static_cast<double>(elapsed_us);
    std::cout << "NPU_E2E_OK mode=" << (shared ? "shared" : "exclusive")
              << " workload=" << (shared ? workload : "direct") << " share=" << (shared ? share : "1000")
              << " runs=" << latencies.size() << " elapsed_us=" << elapsed_us << " throughput_fps=" << throughput_fps
              << " average_latency_us=" << average_latency_us << " p95_latency_us=" << p95_latency_us
              << " max_latency_us=" << max_latency_us << '\n';
    return 0;
}
