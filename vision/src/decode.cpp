#include "common.hpp"
namespace vision {
static float distance(const float *v) {
  float m = *std::max_element(v, v + 16), sum = 0, weighted = 0;
  for (int i = 0; i < 16; ++i) {
    float e = std::exp(v[i] - m);
    sum += e;
    weighted += i * e;
  }
  return weighted / sum;
}
static std::vector<int> nms(const std::vector<Candidate> &c) {
  std::vector<int> order(c.size()), keep;
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return c[a].score > c[b].score; });
  for (int i : order) {
    bool suppressed = false;
    for (int j : keep)
      if (c[i].label == c[j].label) {
        float intersection = (c[i].box & c[j].box).area(),
              uni = c[i].box.area() + c[j].box.area() - intersection;
        if (uni > 0 && intersection / uni > 0.7f) {
          suppressed = true;
          break;
        }
      }
    if (!suppressed) {
      keep.push_back(i);
      if (keep.size() == 100)
        break;
    }
  }
  return keep;
}
Json decode(const std::string &kind, const std::vector<Tensor> &t,
            const Transform &tr) {
  Json out = {{"kind", kind}, {"objects", Json::array()}};
  if (kind == "detection") {
    require(t.size() == 1, "detection requires one Hailo NMS output");
    // HAILO_NMS_BY_CLASS uses a count followed by that class's actual boxes,
    // NOT a fixed maximum-size stride between classes.
    const auto &v = t[0].values;
    size_t offset = 0;
    for (int label = 0; label < 80; ++label) {
      require(offset < v.size(), "truncated NMS counts");
      float raw = v[offset++];
      require(std::isfinite(raw) && raw >= 0 && raw <= 100 &&
                  raw == std::floor(raw),
              "invalid NMS count");
      int n = raw;
      require(offset + size_t(n) * 5 <= v.size(), "truncated NMS boxes");
      for (int i = 0; i < n; ++i) {
        float y0 = v[offset], x0 = v[offset + 1], y1 = v[offset + 2],
              x1 = v[offset + 3], score = v[offset + 4];
        offset += 5;
        if (score < 0.3f)
          continue;
        out["objects"].push_back(
            {{"class_id", label},
             {"score", score},
             {"box",
              tr.box({x0 * 640, y0 * 640, (x1 - x0) * 640, (y1 - y0) * 640})}});
      }
    }
    return out;
  }
  require(kind == "segmentation" || kind == "pose", "unknown adapter");
  require(t.size() == (kind == "pose" ? 9 : 10), "unexpected raw head count");
  for (const auto &a : t)
    require(a.h > 0 && a.w > 0 && a.c > 0 &&
                a.values.size() == size_t(a.h) * a.w * a.c,
            "truncated raw tensor");
  const Tensor *proto = nullptr;
  std::vector<Candidate> candidates;
  for (const auto &a : t)
    if (a.h == 160 && a.w == 160 && a.c == 32)
      proto = &a;
  for (int stride : {8, 16, 32}) {
    int side = 640 / stride;
    const Tensor *boxes = nullptr, *scores = nullptr, *extras = nullptr;
    for (const auto &a : t)
      if (a.h == side && a.w == side) {
        if (a.c == 64)
          boxes = &a;
        if (a.c == (kind == "pose" ? 1 : 80))
          scores = &a;
        if (a.c == (kind == "pose" ? 51 : 32))
          extras = &a;
      }
    require(boxes && scores && extras,
            "incompatible YOLOv8 tensor layout at stride " +
                std::to_string(stride));
    for (int y = 0; y < side; ++y)
      for (int x = 0; x < side; ++x) {
        const float *s = scores->at(y, x);
        int label = std::max_element(s, s + scores->c) - s;
        float score = s[label];
        // The pinned HEFs include sigmoid on class scores, while keypoint
        // visibility is raw.
        if (score < 0.3f || !std::isfinite(score))
          continue;
        const float *b = boxes->at(y, x), *e = extras->at(y, x);
        float cx = (x + 0.5f) * stride, cy = (y + 0.5f) * stride;
        float l = distance(b) * stride, top = distance(b + 16) * stride,
              r = distance(b + 32) * stride, bottom = distance(b + 48) * stride;
        candidates.push_back({{cx - l, cy - top, l + r, top + bottom},
                              score,
                              label,
                              {e, e + extras->c},
                              x,
                              y,
                              stride});
      }
  }
  if (kind == "segmentation")
    require(proto, "missing mask prototypes");
  for (int i : nms(candidates)) {
    const auto &c = candidates[i];
    Json object = {
        {"class_id", c.label}, {"score", c.score}, {"box", tr.box(c.box)}};
    if (kind == "pose") {
      object["keypoints"] = Json::array();
      for (int k = 0; k < 17; ++k)
        object["keypoints"].push_back(
            {tr.x((c.extra[k * 3] * 2 + c.x) * c.stride),
             tr.y((c.extra[k * 3 + 1] * 2 + c.y) * c.stride),
             sigmoid(c.extra[k * 3 + 2])});
    } else {
      cv::Mat logits(160, 160, CV_32F);
      for (int y = 0; y < 160; ++y)
        for (int x = 0; x < 160; ++x) {
          const float *p = proto->at(y, x);
          float sum = 0;
          for (int k = 0; k < 32; ++k)
            sum += p[k] * c.extra[k];
          logits.at<float>(y, x) = sum;
        }
      cv::Mat full;
      cv::resize(logits, full, {640, 640}, 0, 0, cv::INTER_LINEAR);
      cv::Mat mask;
      cv::resize(full(cv::Rect(tr.left, tr.top, tr.resized_w, tr.resized_h)),
                 mask, {tr.width, tr.height});
      const auto b = tr.box(c.box);
      int x0 = std::floor(b[0].get<float>()),
          y0 = std::floor(b[1].get<float>());
      int x1 = std::ceil(b[2].get<float>()), y1 = std::ceil(b[3].get<float>());
      // Row-major foreground [start,length] runs, cropped to the predicted box.
      Json runs = Json::array();
      int start = -1, length = 0;
      for (int y = 0; y < tr.height; ++y)
        for (int x = 0; x < tr.width; ++x) {
          bool on = x >= x0 && x < x1 && y >= y0 && y < y1 &&
                    mask.at<float>(y, x) > 0;
          if (on) {
            if (start < 0)
              start = y * tr.width + x;
            ++length;
          } else if (start >= 0) {
            runs.push_back({start, length});
            start = -1;
            length = 0;
          }
        }
      if (start >= 0)
        runs.push_back({start, length});
      object["mask"] = {
          {"width", tr.width}, {"height", tr.height}, {"runs", runs}};
    }
    out["objects"].push_back(std::move(object));
  }
  return out;
}
} // namespace vision
