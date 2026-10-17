#pragma once
#define GL_GLEXT_PROTOTYPES
#include "common.hpp"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
namespace vision {
class Viewer {
  SDL_Window *window = nullptr;
  SDL_GLContext context = nullptr;
  int width = 1280, height = 800;
  void texture(const cv::Mat &image, float x, float y, float w, float h) {
    GLuint id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    auto format = image.channels() == 4 ? GL_RGBA : GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, format, image.cols, image.rows, 0, format,
                 GL_UNSIGNED_BYTE, image.data);
    glEnable(GL_TEXTURE_2D);
    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(x, y);
    glTexCoord2f(1, 0);
    glVertex2f(x + w, y);
    glTexCoord2f(1, 1);
    glVertex2f(x + w, y + h);
    glTexCoord2f(0, 1);
    glVertex2f(x, y + h);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glDeleteTextures(1, &id);
  }

public:
  std::string renderer, vendor;
  explicit Viewer(bool allow_software) {
    require(SDL_Init(SDL_INIT_VIDEO) == 0, SDL_GetError());
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    window =
        SDL_CreateWindow("OSH vision: original and active tasks",
                         SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width,
                         height, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    require(window, SDL_GetError());
    context = SDL_GL_CreateContext(window);
    require(context, SDL_GetError());
    const char *r = reinterpret_cast<const char *>(glGetString(GL_RENDERER)),
               *v = reinterpret_cast<const char *>(glGetString(GL_VENDOR));
    require(r && v, "no GL renderer");
    renderer = r;
    vendor = v;
    std::string lower = renderer;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    bool software = lower.find("softpipe") != std::string::npos ||
                    lower.find("llvmpipe") != std::string::npos ||
                    lower.find("software") != std::string::npos ||
                    lower.find("swrast") != std::string::npos;
    require(allow_software || !software,
            "hardware rendering required; GL renderer=" + renderer);
    SDL_GL_SetSwapInterval(0);
    std::cout << Json{{"event", "renderer"},
                      {"vendor", vendor},
                      {"renderer", renderer},
                      {"hardware", !software}}
                     .dump()
              << std::endl;
  }
  ~Viewer() {
    if (context)
      SDL_GL_DeleteContext(context);
    if (window)
      SDL_DestroyWindow(window);
    SDL_Quit();
  }
  bool begin() {
    SDL_Event e;
    while (SDL_PollEvent(&e))
      if (e.type == SDL_QUIT ||
          (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE))
        return false;
    SDL_GL_GetDrawableSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, height, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(.06f, .06f, .06f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    return true;
  }
  void tile(const cv::Mat &rgb, const Json &result, const std::string &label,
            int index, int count) {
    int columns = std::ceil(std::sqrt(count)),
        rows = (count + columns - 1) / columns;
    float tw = float(width) / columns, th = float(height) / rows,
          ox = (index % columns) * tw, oy = (index / columns) * th;
    cv::Mat text(28, int(tw), CV_8UC3, cv::Scalar(24, 24, 24));
    cv::putText(text, label, {8, 20}, cv::FONT_HERSHEY_SIMPLEX, .45,
                {230, 230, 230}, 1, cv::LINE_AA);
    texture(text, ox, oy, tw, 28);
    if (rgb.empty())
      return;
    float scale = std::min(tw / rgb.cols, (th - 30) / rgb.rows),
          w = rgb.cols * scale, h = rgb.rows * scale, x = ox + (tw - w) / 2,
          y = oy + 30;
    texture(rgb, x, y, w, h);
    if (!result.contains("objects"))
      return;
    const int edges[][2] = {{0, 1},   {0, 2},   {1, 3},   {2, 4},
                            {5, 6},   {5, 7},   {7, 9},   {6, 8},
                            {8, 10},  {5, 11},  {6, 12},  {11, 12},
                            {11, 13}, {13, 15}, {12, 14}, {14, 16}};
    for (const auto &o : result["objects"]) {
      if (o.contains("mask")) {
        const auto &m = o["mask"];
        require(m.at("width") == rgb.cols && m.at("height") == rgb.rows,
                "mask size mismatch");
        cv::Mat rgba(rgb.rows, rgb.cols, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        for (const auto &run : m.at("runs")) {
          int start = run[0], n = run[1];
          require(start >= 0 && n >= 0 && size_t(start) + n <= rgba.total(),
                  "invalid mask run");
          for (int i = start; i < start + n; ++i)
            reinterpret_cast<cv::Vec4b *>(rgba.data)[i] = {40, 220, 100, 100};
        }
        texture(rgba, x, y, w, h);
      }
      const auto &b = o.at("box");
      glColor4f(.2f, 1, .2f, 1);
      glLineWidth(2);
      glBegin(GL_LINE_LOOP);
      glVertex2f(x + b[0].get<float>() * scale, y + b[1].get<float>() * scale);
      glVertex2f(x + b[2].get<float>() * scale, y + b[1].get<float>() * scale);
      glVertex2f(x + b[2].get<float>() * scale, y + b[3].get<float>() * scale);
      glVertex2f(x + b[0].get<float>() * scale, y + b[3].get<float>() * scale);
      glEnd();
      if (o.contains("keypoints")) {
        const auto &p = o["keypoints"];
        require(p.size() == 17, "expected COCO-17 keypoints");
        glColor4f(1, .6f, .1f, 1);
        glBegin(GL_LINES);
        for (auto &edge : edges)
          if (p[edge[0]][2] > .5 && p[edge[1]][2] > .5)
            for (int k : edge)
              glVertex2f(x + p[k][0].get<float>() * scale,
                         y + p[k][1].get<float>() * scale);
        glEnd();
        glPointSize(5);
        glBegin(GL_POINTS);
        for (auto &k : p)
          if (k[2] > .5)
            glVertex2f(x + k[0].get<float>() * scale,
                       y + k[1].get<float>() * scale);
        glEnd();
      }
    }
  }
  uint64_t finish(const std::string &screenshot = "") {
    glFinish();
    require(glGetError() == GL_NO_ERROR, "OpenGL rendering failed");
    uint64_t completed = now();
    if (!screenshot.empty()) {
      cv::Mat rgb(height, width, CV_8UC3), bgr;
      glPixelStorei(GL_PACK_ALIGNMENT, 1);
      glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
      cv::flip(rgb, rgb, 0);
      cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
      require(cv::imwrite(screenshot, bgr), "screenshot failed");
    }
    SDL_GL_SwapWindow(window);
    return completed;
  }
};
} // namespace vision
