#include "metrics.hpp"
int main() {
  using namespace vision;
  RollingMetrics m;
  require(m.summary(1000000000ULL)["throughput_fps"].is_null(),"no data must not report FPS");
  m.add(1100000000ULL,{{"inference_ms",10.}},1000000000ULL);
  require(m.summary(1500000000ULL)["throughput_fps"].is_null(),"short window must not extrapolate FPS");
  m.add(2100000000ULL,{{"inference_ms",20.}},2000000000ULL);
  auto s=m.summary(3000000000ULL);
  require(s["throughput_fps"]==1.,"wall-clock frame rate");
  require(s["latency_ms"]["inference_ms"]["mean"]==15.,"mean");
  require(s["latency_ms"]["inference_ms"]["p95"]==20.,"nearest-rank P95");
  require(m.summary(3000000000ULL)==s,"render refresh must not add samples");
  s=m.summary(8000000000ULL);
  require(s["throughput_fps"]==0. && s["samples"]==0,"idle tasks must decay to zero");
  require(s["latency_ms"].empty(),"expired latency must not remain current");
  RollingMetrics reconnected;
  require(reconnected.summary(9000000000ULL)["samples"]==0,"connection stats reset");
  std::cout<<"VISION_METRICS_OK\n";
}
