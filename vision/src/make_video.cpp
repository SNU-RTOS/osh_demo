#include "common.hpp"
int main(int argc, char **argv) {
  try {
    using namespace vision;
    require(argc == 3, "usage: vision_make_video IMAGE OUTPUT.mp4");
    auto image = cv::imread(argv[1]);
    require(!image.empty(), "cannot read source image");
    cv::VideoWriter writer(argv[2], cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           10, {960, 640});
    require(writer.isOpened(), "cannot create video");
    for (int i = 0; i < 100; ++i) {
      double t = (1 - std::cos(i * 2 * 3.141592653589793 / 100)) / 2,
             zoom = 1 + .15 * t;
      int w = image.cols / zoom, h = std::min(int(w * 2. / 3), image.rows);
      w = std::min(int(h * 1.5), image.cols);
      int x = (image.cols - w) * t, y = (image.rows - h) / 2;
      cv::Mat f;
      cv::resize(image(cv::Rect(x, y, w, h)), f, {960, 640});
      writer.write(f);
    }
    std::cout << "VIDEO_READY " << argv[2] << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
