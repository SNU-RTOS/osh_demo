#include "common.hpp"
#ifndef VISION_MOCK_ONLY
#include <hailo/hailort.hpp>
#endif
#include <csignal>
#include <thread>
namespace vision {
static volatile std::sig_atomic_t stopped = 0;
static void stop(int) { stopped = 1; }
struct InferenceError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
#ifndef VISION_MOCK_ONLY
class Model {
  std::unique_ptr<hailort::VDevice> device;
  std::shared_ptr<hailort::ConfiguredNetworkGroup> network;
  std::unique_ptr<hailort::InferVStreams> streams;
  std::map<std::string, hailort::MemoryView> input_views, output_views;
  std::vector<uint8_t> input;
  std::vector<Tensor> outputs;

public:
  std::string id;
  explicit Model(const std::string &path, const std::string &kind) {
    auto ids = hailort::Device::scan();
    require(bool(ids) && ids->size() == 1,
            "worker requires exactly one assigned device");
    id = ids->front();
    auto d = hailort::VDevice::create(std::vector<std::string>{id});
    require(bool(d), "VDevice failed");
    device = d.release();
    auto hef = hailort::Hef::create(path);
    require(bool(hef), "invalid HEF: " + path);
    auto params = device->create_configure_params(hef.value());
    require(bool(params), "configure params failed");
    auto networks = device->configure(hef.value(), params.value());
    require(bool(networks) && networks->size() == 1,
            "requires one network group");
    network = networks->front();
    auto ip = network->make_input_vstream_params(
        {}, HAILO_FORMAT_TYPE_UINT8, 2000, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    auto op = network->make_output_vstream_params(
        {}, HAILO_FORMAT_TYPE_FLOAT32, 2000, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    require(bool(ip) && bool(op), "vstream parameters failed");
    if (kind != "detection")
      for (auto &[name, param] : op.value())
        param.user_buffer_format.order = HAILO_FORMAT_ORDER_NHWC;
    auto s = hailort::InferVStreams::create(*network, ip.value(), op.value());
    require(bool(s), "InferVStreams failed");
    streams = std::make_unique<hailort::InferVStreams>(s.release());
    require(streams->get_input_vstreams().size() == 1,
            "requires one RGB input");
    auto &in = streams->get_input_vstreams().front().get();
    const auto info = in.get_info();
    require(info.shape.height == 640 && info.shape.width == 640 &&
                info.shape.features == 3 &&
                in.get_frame_size() == 640 * 640 * 3,
            "requires 640x640 NHWC RGB UINT8 input");
    input.resize(in.get_frame_size());
    input_views.emplace(in.name(),
                        hailort::MemoryView(input.data(), input.size()));
    outputs.reserve(streams->get_output_vstreams().size());
    for (auto &ref : streams->get_output_vstreams()) {
      auto &v = ref.get();
      const auto vi = v.get_info();
      if (kind == "detection")
        require(vi.format.order == HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS &&
                    vi.nms_shape.number_of_classes == 80 &&
                    vi.nms_shape.max_bboxes_per_class == 100,
                "detection requires the pinned 80-class Hailo NMS model");
      // HailoRT transforms native FCR/quantized device tensors to requested
      // NHWC FLOAT32 buffers.
      outputs.push_back(
          {v.name(), int(vi.shape.height), int(vi.shape.width),
           int(vi.shape.features),
           std::vector<float>(v.get_frame_size() / sizeof(float))});
      auto &buffer = outputs.back().values;
      output_views.emplace(
          v.name(),
          hailort::MemoryView(buffer.data(), buffer.size() * sizeof(float)));
      std::cerr << "TENSOR " << v.name() << " " << vi.shape.height << "x"
                << vi.shape.width << "x" << vi.shape.features << "\n";
    }
  }
  const std::vector<Tensor> &infer(const cv::Mat &rgb) {
    std::memcpy(input.data(), rgb.data, input.size());
    auto status = streams->infer(input_views, output_views, 1);
    if (status != HAILO_SUCCESS)
      throw InferenceError("inference failed status=" + std::to_string(status));
    return outputs;
  }
};
#endif
int worker(int argc, char **argv) {
  std::string socket_path =
                  env("OSH_VISION_SOCKET", "/run/osh-vision/hub.sock"),
              task = env("OSH_VISION_TASK", "local"), kind = "detection",
              model = env("MODEL_PATH");
  bool mock = false;
  int delay = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--mock")
      mock = true;
    else if (a == "--kind" && i + 1 < argc)
      kind = argv[++i];
    else if (a == "--model" && i + 1 < argc)
      model = argv[++i];
    else if (a == "--socket" && i + 1 < argc)
      socket_path = argv[++i];
    else if (a == "--task" && i + 1 < argc)
      task = argv[++i];
    else if (a == "--delay-ms" && i + 1 < argc)
      delay = std::stoi(argv[++i]);
    else
      throw std::runtime_error(
          "usage: vision_worker --kind detection|segmentation|pose [--model "
          "HEF] [--socket PATH] [--task ID] [--mock] [--delay-ms N]");
  }
  require(kind == "detection" || kind == "segmentation" || kind == "pose",
          "invalid feature kind");
  require(delay >= 0 && delay <= 10000, "invalid mock delay");
  std::signal(SIGTERM, stop);
  std::signal(SIGINT, stop);
  std::string device = "mock";
#ifndef VISION_MOCK_ONLY
  std::unique_ptr<Model> engine;
  if (!mock) {
    require(env("NPU_COUNT", "1") == "1", "vision requires NPU_COUNT=1");
    require(env("NPU_SHARE").empty(), "shared visual inference is deferred");
    engine = std::make_unique<Model>(model, kind);
    device = engine->id;
  }
#else
  require(mock, "this test binary supports --mock only");
#endif
  while (!stopped) {
    int fd = connect_socket(socket_path);
    if (fd < 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      continue;
    }
    timeout(fd, 5);
    try {
      require(send(fd, {{"type", "hello"},
                        {"task", task},
                        {"kind", kind},
                        {"model", model},
                        {"device", device},
                        {"uid", env("OSH_VISION_UID", task)},
                        {"mock", mock}}),
              "registration send failed");
      Packet p;
      while (!stopped && receive(fd, p)) {
        require(p.meta.at("type") == "frame", "expected frame");
        auto meta = p.meta;
        int w = meta.at("width"), h = meta.at("height");
        require(w > 0 && w <= 1920 && h > 0 && h <= 1080 &&
                    p.data.size() == size_t(w) * h * 3,
                "invalid RGB frame");
        meta["received_ns"] = now();
        Transform tr{w, h};
        cv::Mat rgb(h, w, CV_8UC3, p.data.data());
        meta["preprocess_start_ns"] = now();
        auto input = tr.preprocess(rgb);
        meta["preprocess_end_ns"] = now();
        meta["inference_start_ns"] = now();
        Json result;
        if (mock) {
          std::this_thread::sleep_for(std::chrono::milliseconds(delay));
          result = {
              {"kind", kind},
              {"objects",
               Json::array({{{"class_id", 0},
                             {"score", 1.0},
                             {"box", {w * .2, h * .2, w * .8, h * .8}}}})}};
        }
#ifndef VISION_MOCK_ONLY
        const std::vector<Tensor> *tensors = nullptr;
        if (!mock)
          tensors = &engine->infer(input);
#endif
        meta["inference_end_ns"] = now();
#ifndef VISION_MOCK_ONLY
        if (!mock)
          result = decode(kind, *tensors, tr);
#endif
        meta["postprocess_end_ns"] = now();
        meta["type"] = "result";
        meta["result"] = result;
        meta["uid"] = env("OSH_VISION_UID", task);
        meta["task"] = task;
        meta["model"] = model;
        meta["device"] = device;
        meta["mock"] = mock;
        meta["warmup"] = meta.at("task_frame").get<int>() <= 5;
        meta["result_send_ns"] = now();
        require(send(fd, meta), "result send failed");
        auto timing = meta;
        timing["object_count"] = result["objects"].size();
        timing.erase("result");
        std::cout << Json{{"event", "inference"}, {"timing", timing}}.dump()
                  << std::endl;
        Packet ack;
        require(receive(fd, ack), "missing output acknowledgment");
        require(ack.meta.at("type") == "ack" &&
                    ack.meta.at("session") == meta.at("session") &&
                    ack.meta.at("seq") == meta.at("seq"),
                "mismatched output acknowledgment");
        std::cout << ack.meta.dump() << std::endl;
      }
    } catch (const InferenceError &) {
      close(fd);
      throw;
    } catch (const std::exception &e) {
      std::cerr << "WORKER_SESSION_ERROR " << e.what() << std::endl;
    }
    close(fd);
  }
  return 0;
}
} // namespace vision
int main(int argc, char **argv) {
  try {
    return vision::worker(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "WORKER_FAILED " << e.what() << "\n";
    return 1;
  }
}
