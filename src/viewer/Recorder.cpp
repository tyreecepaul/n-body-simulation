#include "Recorder.h"

#include <GL/glew.h>

bool Recorder::start(const std::string &path, int widthPx, int heightPx,
                     int fps) {
  stop();
  // Raw RGB in; OpenGL rows are bottom-up, so flip; x264 needs even sizes.
  const std::string command =
      "ffmpeg -y -loglevel error -f rawvideo -pix_fmt rgb24 -s " +
      std::to_string(widthPx) + "x" + std::to_string(heightPx) +
      " -framerate " + std::to_string(fps) +
      " -i - -vf \"vflip,crop=trunc(iw/2)*2:trunc(ih/2)*2\" -c:v libx264"
      " -preset veryfast -crf 18 -pix_fmt yuv420p \"" +
      path + "\"";
  pipe_ = popen(command.c_str(), "w");
  if (!pipe_) {
    return false;
  }
  width_ = widthPx;
  height_ = heightPx;
  frames_ = 0;
  path_ = path;
  pixels_.resize(static_cast<std::size_t>(widthPx) * heightPx * 3);
  return true;
}

void Recorder::capture() {
  if (!pipe_) {
    return;
  }
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadBuffer(GL_BACK);
  glReadPixels(0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE,
               pixels_.data());
  if (std::fwrite(pixels_.data(), 1, pixels_.size(), pipe_) !=
      pixels_.size()) {
    stop(); // ffmpeg went away
    return;
  }
  ++frames_;
}

void Recorder::stop() {
  if (pipe_) {
    pclose(pipe_);
    pipe_ = nullptr;
  }
}
