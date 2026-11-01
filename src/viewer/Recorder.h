#pragma once
// Records the window to an MP4 by piping raw frames to ffmpeg.
#include <cstdio>
#include <string>
#include <vector>

class Recorder {
public:
  ~Recorder() { stop(); }
  // Starts ffmpeg writing `path` at `fps`; false if it couldn't start.
  bool start(const std::string &path, int widthPx, int heightPx, int fps);
  // Reads the current back buffer and sends it to ffmpeg.
  void capture();
  void stop();

  bool active() const { return pipe_ != nullptr; }
  int width() const { return width_; }
  int height() const { return height_; }
  const std::string &path() const { return path_; }
  long frames() const { return frames_; }

private:
  std::FILE *pipe_ = nullptr;
  int width_ = 0, height_ = 0;
  long frames_ = 0;
  std::string path_;
  std::vector<unsigned char> pixels_;
};
