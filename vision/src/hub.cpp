#include "viewer.hpp"
#include <condition_variable>
#include <csignal>
#include <fcntl.h>
#include <fstream>
#include <sys/file.h>
#include <thread>
namespace vision {
static volatile std::sig_atomic_t stopped = 0;
static void stop(int) { stopped = 1; }
struct Frame {
  Json meta;
  std::vector<uint8_t> data;
  cv::Mat rgb;
};
struct Peer {
  int fd = -1;
  std::atomic<bool> live{true};
  std::atomic<uint64_t> disconnected_ns{0};
  std::mutex mutex;
  std::condition_variable cv;
  std::shared_ptr<Frame> pending, completed_frame, display_frame;
  Json completed = Json::object(), display = Json::object(),
       hello = Json::object();
  bool ready = false, ack_ready = false;
  Json ack;
  uint64_t drops = 0, task_frame = 0;
  std::thread thread;
};
static void serve(const std::shared_ptr<Peer> &p, const std::string &session) {
  try {
    timeout(p->fd, 5);
    Packet packet;
    require(receive(p->fd, packet), "missing hello");
    auto hello = packet.meta;
    std::string task = hello.at("task"), kind = hello.at("kind");
    require(hello.at("type") == "hello" && !task.empty() && task.size() <= 128,
            "invalid registration");
    require(kind == "detection" || kind == "segmentation" || kind == "pose",
            "invalid task kind");
    {
      std::lock_guard<std::mutex> l(p->mutex);
      p->hello = hello;
    }
    std::cout << Json{{"event", "connected"},
                      {"task", task},
                      {"session", session},
                      {"kind", kind}}
                     .dump()
              << std::endl;
    while (!stopped && p->live) {
      std::shared_ptr<Frame> f;
      Json meta;
      {
        std::unique_lock<std::mutex> l(p->mutex);
        p->cv.wait_for(l, std::chrono::milliseconds(200),
                       [&] { return stopped || !p->live || p->pending; });
        if (stopped || !p->live)
          break;
        if (!p->pending)
          continue;
        f = std::move(p->pending);
        meta = f->meta;
        meta["drops"] = p->drops;
        meta["task_frame"] = ++p->task_frame;
      }
      meta["input_send_ns"] = now();
      require(send(p->fd, meta, f->data), "frame send failed");
      require(receive(p->fd, packet), "result timeout/disconnect");
      require(packet.meta.at("type") == "result" &&
                  packet.meta.at("session") == session &&
                  packet.meta.at("seq") == meta.at("seq") &&
                  packet.meta.at("task") == task,
              "result identity mismatch");
      // Source-owned identity/timing fields cannot be replaced by a client.
      for (const char *key :
           {"session", "seq", "width", "height", "acquired_ns", "decode_end_ns",
            "media_pts_ms", "input_send_ns", "drops", "task_frame"})
        packet.meta[key] = meta.at(key);
      packet.meta["result_received_ns"] = now();
      {
        std::unique_lock<std::mutex> l(p->mutex);
        p->completed = std::move(packet.meta);
        p->completed_frame = f;
        p->ready = true;
        p->ack_ready = false;
        p->cv.wait(l, [&] { return stopped || !p->live || p->ack_ready; });
        if (stopped || !p->live)
          break;
        auto ack = p->ack;
        l.unlock();
        require(send(p->fd, ack), "ack send failed");
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "PEER_DISCONNECTED " << e.what() << "\n";
  }
  p->disconnected_ns = now();
  p->live = false;
  shutdown(p->fd, SHUT_RDWR);
  p->cv.notify_all();
}
static Json measured(Json m, uint64_t render_start, uint64_t done,
                     bool headless) {
  Json d;
  auto delta = [&](const char *name, const char *a, const char *b) {
    uint64_t x = m.at(a), y = m.at(b);
    require(y >= x, "non-monotonic timing");
    d[name] = (y - x) / 1e6;
  };
  delta("decode_ms", "acquired_ns", "decode_end_ns");
  delta("queue_ms", "decode_end_ns", "input_send_ns");
  delta("input_transfer_ms", "input_send_ns", "received_ns");
  delta("preprocess_ms", "preprocess_start_ns", "preprocess_end_ns");
  delta("inference_ms", "inference_start_ns", "inference_end_ns");
  delta("postprocess_ms", "inference_end_ns", "postprocess_end_ns");
  delta("result_transfer_ms", "result_send_ns", "result_received_ns");
  delta("acquisition_to_result_ms", "acquired_ns", "result_received_ns");
  m["render_start_ns"] = render_start;
  m["output_applied_ns"] = done;
  m["displayed"] = !headless;
  m["gpu_complete_ns"] = headless ? Json(nullptr) : Json(done);
  if (!headless) {
    delta("render_queue_ms", "result_received_ns", "render_start_ns");
    delta("render_ms", "render_start_ns", "gpu_complete_ns");
    delta("acquisition_to_gpu_ms", "acquired_ns", "gpu_complete_ns");
  }
  m["durations"] = d;
  m["event"] = "frame";
  return m;
}
int hub(int argc, char **argv) {
  std::string source = "vision/assets/people.jpg",
              socket_path = "/tmp/osh-vision/hub.sock",
              output = "vision/results", screenshot;
  double fps = 10;
  int frames = 0;
  bool headless = false, allow_software = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--headless")
      headless = true;
    else if (a == "--allow-software")
      allow_software = true;
    else if (a == "--source" && i + 1 < argc)
      source = argv[++i];
    else if (a == "--socket" && i + 1 < argc)
      socket_path = argv[++i];
    else if (a == "--output" && i + 1 < argc)
      output = argv[++i];
    else if (a == "--fps" && i + 1 < argc)
      fps = std::stod(argv[++i]);
    else if (a == "--frames" && i + 1 < argc)
      frames = std::stoi(argv[++i]);
    else if (a == "--screenshot" && i + 1 < argc)
      screenshot = argv[++i];
    else
      throw std::runtime_error(
          "usage: vision_hub --source IMAGE_OR_VIDEO [--fps 10] [--frames N] "
          "[--socket PATH] [--output DIR] [--headless] [--screenshot PNG]");
  }
  require(fps > 0 && fps <= 120 && frames >= 0,
          "invalid replay rate/frame count");
  require(socket_path.size() < sizeof(sockaddr_un::sun_path),
          "socket path too long");
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);
  std::filesystem::create_directories(output);
  auto parent = std::filesystem::path(socket_path).parent_path();
  std::filesystem::create_directories(parent);
  int lock =
      open((socket_path + ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
  require(lock >= 0 && flock(lock, LOCK_EX | LOCK_NB) == 0,
          "session already running");
  std::unique_ptr<Viewer> viewer;
  if (!headless)
    viewer = std::make_unique<Viewer>(allow_software);
  int listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  require(listener >= 0, "listener failed");
  unlink(socket_path.c_str());
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strcpy(address.sun_path, socket_path.c_str());
  require(bind(listener, reinterpret_cast<sockaddr *>(&address),
               sizeof address) == 0 &&
              listen(listener, 16) == 0,
          "bind/listen failed");
  std::string session = std::to_string(now());
  std::vector<std::shared_ptr<Peer>> peers;
  struct Cleanup {
    std::vector<std::shared_ptr<Peer>> &peers;
    int listener, lock;
    std::string socket_path;
    ~Cleanup() {
      stopped = 1;
      for (auto &p : peers) {
        p->live = false;
        p->cv.notify_all();
        shutdown(p->fd, SHUT_RDWR);
      }
      for (auto &p : peers) {
        if (p->thread.joinable())
          p->thread.join();
        close(p->fd);
      }
      close(listener);
      unlink(socket_path.c_str());
      close(lock);
    }
  } cleanup{peers, listener, lock, socket_path};
  std::ofstream log(output + "/frames.jsonl", std::ios::app);
  require(bool(log), "cannot open metrics log");
  log << Json{{"event", "session"},
              {"session", session},
              {"headless", headless},
              {"renderer", viewer ? viewer->renderer : "none"},
              {"source", source},
              {"fps", fps}}
             .dump()
      << std::endl;
  cv::Mat still = cv::imread(source), latest;
  cv::VideoCapture video;
  bool image = !still.empty();
  if (!image)
    require(video.open(source), "cannot open input " + source);
  uint64_t seq = 0, next = now(), drain = 0;
  std::shared_ptr<Frame> last;
  std::cout << Json{{"event", "listening"},
                    {"socket", socket_path},
                    {"session", session}}
                   .dump()
            << std::endl;
  while (!stopped) {
    int client = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
    if (client >= 0) {
      size_t active = std::count_if(peers.begin(), peers.end(),
                                    [](auto &p) { return p->live.load(); });
      if (active >= 8)
        close(client);
      else {
        auto p = std::make_shared<Peer>();
        p->fd = client;
        p->thread = std::thread(serve, p, session);
        peers.push_back(p);
      }
    }
    uint64_t tick = now();
    if (tick >= next && (frames == 0 || seq < uint64_t(frames))) {
      auto f = std::make_shared<Frame>();
      uint64_t acquired = now();
      cv::Mat bgr;
      if (image) {
        bgr = cv::imread(source);
        require(!bgr.empty(), "image decode failed");
      } else if (!video.read(bgr)) {
        video.set(cv::CAP_PROP_POS_FRAMES, 0);
        require(video.read(bgr), "video replay failed");
      }
      // Bound transport payload and retained per-task source images.
      double ratio = std::min(1., std::min(1920. / bgr.cols, 1080. / bgr.rows));
      if (ratio < 1)
        cv::resize(bgr, bgr, {}, ratio, ratio);
      cv::cvtColor(bgr, f->rgb, cv::COLOR_BGR2RGB);
      f->data = pixels(f->rgb);
      f->meta = {
          {"type", "frame"},
          {"session", session},
          {"seq", ++seq},
          {"width", f->rgb.cols},
          {"height", f->rgb.rows},
          {"acquired_ns", acquired},
          {"decode_end_ns", now()},
          {"media_pts_ms", image ? 0. : video.get(cv::CAP_PROP_POS_MSEC)}};
      last = f;
      latest = f->rgb;
      for (auto &p : peers)
        if (p->live) {
          std::lock_guard<std::mutex> l(p->mutex);
          if (p->hello.empty())
            continue;
          if (p->pending) {
            ++p->drops;
            log << Json{{"event", "drop"},
                        {"task", p->hello.at("task")},
                        {"session", session},
                        {"seq", p->pending->meta.at("seq")},
                        {"reason", "pending_replaced"},
                        {"drops", p->drops}}
                       .dump()
                << std::endl;
          }
          p->pending = f;
          p->cv.notify_all();
        }
      next = std::max(next + uint64_t(1e9 / fps), now());
      if (frames && seq == uint64_t(frames))
        drain = now() + 5000000000ULL;
    }
    uint64_t render_start = now();
    if (viewer && !viewer->begin())
      break;
    std::vector<std::shared_ptr<Peer>> tiles;
    for (auto &p : peers) {
      std::lock_guard<std::mutex> l(p->mutex);
      if (!p->hello.empty())
        tiles.push_back(p);
    }
    // Retain a disconnected tile for diagnosis; discard it when the same task
    // reconnects.
    std::map<std::string, std::shared_ptr<Peer>> current;
    for (auto &p : tiles) {
      std::string name = p->hello.at("task");
      auto old = current.find(name);
      if (old != current.end() && old->second->live) {
        old->second->live = false;
        old->second->cv.notify_all();
        shutdown(old->second->fd, SHUT_RDWR);
      }
      current[name] = p;
    }
    tiles.clear();
    for (auto &[name, p] : current)
      tiles.push_back(p);
    if (viewer)
      viewer->tile(latest, Json::object(), "Original", 0, tiles.size() + 1);
    int index = 1;
    std::vector<std::shared_ptr<Peer>> completed;
    for (auto &p : tiles) {
      std::lock_guard<std::mutex> l(p->mutex);
      if (p->ready) {
        p->display = p->completed;
        p->display["render_start_ns"] = now();
        p->display_frame = p->completed_frame;
        p->ready = false;
        completed.push_back(p);
      }
      auto label = p->hello.at("task").get<std::string>() + " / " +
                   p->hello.at("kind").get<std::string>();
      if (!p->live)
        label += " DISCONNECTED";
      else if (!p->display_frame)
        label += " STARTING";
      else if (now() - p->display.at("result_received_ns").get<uint64_t>() >
               2000000000ULL)
        label += " STALE";
      if (p->hello.value("mock", false))
        label += " MOCK";
      if (viewer)
        viewer->tile(p->display_frame ? p->display_frame->rgb : cv::Mat{},
                     p->display.value("result", Json::object()), label, index++,
                     tiles.size() + 1);
    }
    uint64_t done = viewer ? viewer->finish() : now();
    for (auto &p : completed) {
      std::lock_guard<std::mutex> l(p->mutex);
      auto record = measured(p->display, p->display.at("render_start_ns"), done,
                             headless);
      record["renderer"] = viewer ? viewer->renderer : "none";
      log << record.dump() << std::endl;
      record.erase("result");
      record["type"] = "ack";
      p->ack = std::move(record);
      p->ack_ready = true;
      p->cv.notify_all();
    }
    // Reap exited sessions so repeated restarts cannot grow retained state
    // indefinitely.
    for (auto it = peers.begin(); it != peers.end();) {
      auto &p = *it;
      bool superseded = !p->live && !p->hello.empty() &&
                        current[p->hello.at("task")].get() != p.get();
      if (!p->live && (p->hello.empty() || superseded ||
                       (p->disconnected_ns &&
                        now() - p->disconnected_ns > 5000000000ULL))) {
        if (p->thread.joinable())
          p->thread.join();
        close(p->fd);
        it = peers.erase(it);
      } else
        ++it;
    }
    if (drain && now() >= drain) {
      if (viewer && !screenshot.empty()) {
        viewer->begin();
        viewer->tile(latest, Json::object(), "Original", 0, tiles.size() + 1);
        int i = 1;
        for (auto &p : tiles)
          viewer->tile(p->display_frame ? p->display_frame->rgb : cv::Mat{},
                       p->display.value("result", Json::object()),
                       p->hello.at("task"), i++, tiles.size() + 1);
        viewer->finish(screenshot);
      }
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return 0;
}
} // namespace vision
int main(int argc, char **argv) {
  try {
    return vision::hub(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "HUB_FAILED " << e.what() << "\n";
    return 1;
  }
}
