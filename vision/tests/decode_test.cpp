#include "common.hpp"
int main() {
  using namespace vision;
  try {
    Transform tr{1280, 720};
    cv::Mat rgb(720, 1280, CV_8UC3, cv::Scalar(1, 2, 3));
    auto input = tr.preprocess(rgb);
    require(tr.left == 0 && tr.top == 140 && input.rows == 640,
            "letterbox transform");
    // Compact NMS data: two classes, one box each, then empty classes.
    Tensor nms{"nms",
               0,
               0,
               0,
               {1, .25f, .25f, .75f, .75f, .9f, 1, .3f, .3f, .6f, .6f, .8f}};
    nms.values.resize(nms.values.size() + 78, 0);
    auto detections = decode("detection", {nms}, tr);
    require(detections["objects"].size() == 2,
            "variable-length NMS class traversal");
    require(detections["objects"][1]["class_id"] == 1, "second class identity");
    require(std::abs(detections["objects"][0]["box"][0].get<float>() - 320) <
                .01,
            "unletterbox x");
    bool rejected = false;
    nms.values = {100};
    try {
      decode("detection", {nms}, tr);
    } catch (...) {
      rejected = true;
    }
    require(rejected, "truncated NMS accepted");
    for (auto kind : {"pose", "segmentation"}) {
      std::vector<Tensor> tensors;
      for (int side : {80, 40, 20})
        for (int c : {64, kind == std::string("pose") ? 1 : 80,
                      kind == std::string("pose") ? 51 : 32})
          tensors.push_back(
              {"raw", side, side, c, std::vector<float>(side * side * c, 0)});
      if (kind == std::string("segmentation"))
        tensors.push_back(
            {"proto", 160, 160, 32, std::vector<float>(160 * 160 * 32, 1)});
      // One high-confidence candidate; uniform DFL has expected distance 7.5
      // cells.
      tensors[1].values[(40 * 80 + 40) * tensors[1].c] = .9f;
      tensors[2].values[(40 * 80 + 40) * tensors[2].c] = 1;
      auto result = decode(kind, tensors, tr);
      require(result["objects"].size() == 1, "raw head decode");
      if (kind == std::string("pose"))
        require(result["objects"][0]["keypoints"].size() == 17,
                "pose keypoints");
      else
        require(!result["objects"][0]["mask"]["runs"].empty(),
                "segmentation mask missing");
    }
    std::cout << "VISION_DECODE_OK\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
