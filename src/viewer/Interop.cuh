#pragma once
// The OpenGL vertex buffer the viewer draws from, filled directly from the
// simulation's device arrays through CUDA-OpenGL interop: bodies never
// leave the GPU. Requires a current OpenGL context.
#include "DeviceBodies.cuh"
#include <cstddef>

// Whole-system figures gathered while filling the buffer (for the camera's
// follow mode and the colour scale).
struct BodyStats {
  float comX = 0.0f, comY = 0.0f; // centre of mass
  float maxSpeed = 1.0f;
};

class GlBodyBuffer {
public:
  GlBodyBuffer() = default;
  ~GlBodyBuffer();
  GlBodyBuffer(const GlBodyBuffer &) = delete;
  GlBodyBuffer &operator=(const GlBodyBuffer &) = delete;

  // (Re)creates the buffer for n bodies; one vertex per body,
  // {x, y, speed, radius} as 4 floats.
  void resize(std::size_t n);
  // Writes every body's vertex and the stats. Synchronises with the GPU.
  void fill(const DeviceBodies &b);

  unsigned vbo() const { return vbo_; }
  std::size_t size() const { return n_; }
  const BodyStats &stats() const { return stats_; }

private:
  void release();

  unsigned vbo_ = 0;
  struct cudaGraphicsResource *resource_ = nullptr;
  std::size_t n_ = 0;
  DeviceBuffer<float> sums_; // m*x, m*y, m, max speed (as int bits)
  BodyStats stats_;
};
