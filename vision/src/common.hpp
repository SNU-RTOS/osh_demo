#pragma once
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace vision {
using Json = nlohmann::json;
inline uint64_t now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
inline std::string env(const char *key, std::string fallback = "") {
  const char *p = std::getenv(key);
  return p ? p : fallback;
}
inline void require(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error(message);
}
constexpr uint32_t version = 1, max_json = 8 * 1024 * 1024,
                   max_blob = 16 * 1024 * 1024;
struct Packet {
  Json meta;
  std::vector<uint8_t> data;
};
inline bool transfer(int fd, void *p, size_t n, bool write) {
  auto b = static_cast<uint8_t *>(p);
  while (n) {
    ssize_t k = write ? ::send(fd, b, n, MSG_NOSIGNAL) : ::recv(fd, b, n, 0);
    if (k < 0 && errno == EINTR)
      continue;
    if (k <= 0)
      return false;
    b += k;
    n -= k;
  }
  return true;
}
inline bool send(int fd, Json meta, const std::vector<uint8_t> &data = {}) {
  meta["version"] = version;
  std::string s = meta.dump();
  require(s.size() <= max_json && data.size() <= max_blob,
          "packet exceeds limit");
  uint32_t h[] = {htonl(s.size()), htonl(data.size())};
  return transfer(fd, h, sizeof h, true) &&
         transfer(fd, s.data(), s.size(), true) &&
         transfer(fd, const_cast<uint8_t *>(data.data()), data.size(), true);
}
inline bool receive(int fd, Packet &p) {
  uint32_t h[2];
  if (!transfer(fd, h, sizeof h, false))
    return false;
  auto j = ntohl(h[0]), b = ntohl(h[1]);
  require(j > 0 && j <= max_json && b <= max_blob, "invalid packet length");
  std::string s(j, '\0');
  p.data.resize(b);
  if (!transfer(fd, s.data(), j, false) ||
      !transfer(fd, p.data.data(), b, false))
    return false;
  p.meta = Json::parse(s);
  require(p.meta.at("version") == version, "unsupported protocol version");
  return true;
}
inline int connect_socket(const std::string &path) {
  require(path.size() < sizeof(sockaddr_un::sun_path), "socket path too long");
  int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  require(fd >= 0, "socket failed");
  sockaddr_un a{};
  a.sun_family = AF_UNIX;
  std::strcpy(a.sun_path, path.c_str());
  if (connect(fd, reinterpret_cast<sockaddr *>(&a), sizeof a) < 0) {
    close(fd);
    return -1;
  }
  return fd;
}
inline void timeout(int fd, int seconds) {
  timeval t{seconds, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &t, sizeof t);
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &t, sizeof t);
}
inline std::vector<uint8_t> pixels(const cv::Mat &m) {
  require(m.isContinuous(), "non-contiguous image");
  return {m.data, m.data + m.total() * m.elemSize()};
}
struct Transform {
  int width, height, input_w = 640, input_h = 640, left = 0, top = 0,
                     resized_w = 0, resized_h = 0;
  float scale = 1;
  cv::Mat preprocess(const cv::Mat &rgb) {
    width = rgb.cols;
    height = rgb.rows;
    scale = std::min(float(input_w) / width, float(input_h) / height);
    resized_w = std::round(width * scale);
    resized_h = std::round(height * scale);
    left = (input_w - resized_w) / 2;
    top = (input_h - resized_h) / 2;
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114)), resized;
    cv::resize(rgb, resized, {resized_w, resized_h});
    resized.copyTo(out(cv::Rect(left, top, resized_w, resized_h)));
    return out;
  }
  float x(float v) const {
    return std::clamp((v - left) / scale, 0.f, float(width));
  }
  float y(float v) const {
    return std::clamp((v - top) / scale, 0.f, float(height));
  }
  Json box(const cv::Rect2f &r) const {
    return {x(r.x), y(r.y), x(r.x + r.width), y(r.y + r.height)};
  }
};
struct Tensor {
  std::string name;
  int h, w, c;
  std::vector<float> values;
  const float *at(int y, int x) const {
    return values.data() + (y * w + x) * c;
  }
};
struct Candidate {
  cv::Rect2f box;
  float score;
  int label;
  std::vector<float> extra;
  int x = 0, y = 0, stride = 0;
};
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
Json decode(const std::string &kind, const std::vector<Tensor> &t,
            const Transform &tr);
} // namespace vision
