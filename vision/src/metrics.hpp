#pragma once
#include "common.hpp"
#include <deque>
#include <array>

namespace vision {
// Five-second wall-clock window; add once per completed, non-warmup frame.
class RollingMetrics {
  struct Sample { uint64_t at; Json durations; };
  std::deque<Sample> samples;
  uint64_t started = 0;
  static constexpr uint64_t window = 5000000000ULL;
  void prune(uint64_t at) {
    while (!samples.empty() && at >= samples.front().at &&
           at - samples.front().at >= window) samples.pop_front();
  }
public:
  void add(uint64_t at, const Json &durations, uint64_t begin) {
    require(at >= begin, "non-monotonic metric sample");
    if (!started) started = begin;
    samples.push_back({at, durations});
    prune(at);
  }
  Json summary(uint64_t at) {
    prune(at);
    double seconds = started && at >= started ? std::min(double(at-started)/1e9, 5.) : 0.;
    Json result{{"window_seconds", 5}, {"observed_seconds", seconds},
                {"as_of_ns", at},
                {"samples", samples.size()}, {"throughput_fps", nullptr},
                {"latency_ms", Json::object()}};
    // Avoid presenting one short inference interval as sustainable throughput.
    if (seconds >= 1.) result["throughput_fps"] = samples.size()/seconds;
    std::map<std::string,std::vector<double>> values;
    for (const auto &s : samples)
      for (auto it=s.durations.begin();it!=s.durations.end();++it) {
        double v=it.value();
        require(std::isfinite(v) && v>=0, "invalid measured duration");
        values[it.key()].push_back(v);
      }
    for (auto &[key,v] : values) {
      double mean=std::accumulate(v.begin(),v.end(),0.)/v.size();
      std::sort(v.begin(),v.end());
      result["latency_ms"][key]={{"mean",mean},{"p95",v[size_t(std::ceil(v.size()*.95))-1]}};
    }
    return result;
  }
};
inline Json worker_durations(const Json &m) {
  Json result;
  for (auto item : {std::array<const char*,3>{"preprocess_ms","preprocess_start_ns","preprocess_end_ns"},
                   {"inference_ms","inference_start_ns","inference_end_ns"},
                   {"postprocess_ms","inference_end_ns","postprocess_end_ns"},
                   {"pipeline_ms","received_ns","postprocess_end_ns"}}) {
    uint64_t a=m.at(item[1]),b=m.at(item[2]);require(b>=a,"non-monotonic worker timing");
    result[item[0]]=(b-a)/1e6;
  }
  return result;
}
} // namespace vision
