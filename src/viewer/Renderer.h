#pragma once
// Draws the bodies as soft round points from the interop vertex buffer.
#include <cstddef>

// World-to-screen mapping: screen = (world - center) * scale, in pixels
// from the viewport centre (y up).
struct Camera {
  float centerX = 0.0f, centerY = 0.0f;
  float scale = 1.0f; // pixels per world unit
};

enum class ColorMode { Speed, Glow };

struct DrawSettings {
  ColorMode mode = ColorMode::Glow;
  float pointScale = 1.0f;  // multiplies each body's true size
  float minPointPx = 1.5f;  // smallest drawn point; smaller bodies fade
  float brightness = 0.6f;
  float speedMax = 60.0f;   // speed shown at the top of the colour map
};

class Renderer {
public:
  bool init(); // compiles shaders; needs a current GL 4.3 context
  ~Renderer();
  void draw(unsigned vbo, std::size_t count, const Camera &camera,
            int widthPx, int heightPx, const DrawSettings &settings);

private:
  unsigned program_ = 0;
  unsigned vao_ = 0;
  int uCenter_ = -1, uScale_ = -1, uViewport_ = -1, uPointScale_ = -1,
      uMinPoint_ = -1, uSpeedMax_ = -1, uMode_ = -1, uBrightness_ = -1;
};
