#include <GL/glew.h> // before cuda_gl_interop.h, which includes GL headers

#include "Interop.cuh"

#include <cuda_gl_interop.h>

#include <algorithm>

namespace {
constexpr int kBlock = 256;

// One thread per body: writes its vertex and adds its share of the stats.
// Float atomics make the sums order-dependent, which is fine for a camera.
__global__ void fillKernel(int n, const float *__restrict__ posX,
                           const float *__restrict__ posY,
                           const float *__restrict__ velX,
                           const float *__restrict__ velY,
                           const float *__restrict__ mass,
                           const float *__restrict__ radius,
                           float4 *__restrict__ vertices,
                           float *__restrict__ sums) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  float mx = 0.0f, my = 0.0f, m = 0.0f, speed = 0.0f;
  if (i < n) {
    const float x = posX[i], y = posY[i];
    speed = sqrtf(velX[i] * velX[i] + velY[i] * velY[i]);
    vertices[i] = make_float4(x, y, speed, radius[i]);
    m = mass[i];
    mx = m * x;
    my = m * y;
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    mx += __shfl_down_sync(0xffffffffu, mx, offset);
    my += __shfl_down_sync(0xffffffffu, my, offset);
    m += __shfl_down_sync(0xffffffffu, m, offset);
    speed = fmaxf(speed, __shfl_down_sync(0xffffffffu, speed, offset));
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&sums[0], mx);
    atomicAdd(&sums[1], my);
    atomicAdd(&sums[2], m);
    // Non-negative floats order the same as their bit patterns.
    atomicMax(reinterpret_cast<int *>(&sums[3]), __float_as_int(speed));
  }
}
} // namespace

GlBodyBuffer::~GlBodyBuffer() { release(); }

void GlBodyBuffer::release() {
  if (resource_) {
    cudaGraphicsUnregisterResource(resource_);
    resource_ = nullptr;
  }
  if (vbo_) {
    glDeleteBuffers(1, &vbo_);
    vbo_ = 0;
  }
  n_ = 0;
}

void GlBodyBuffer::resize(std::size_t n) {
  if (n == n_ && vbo_) {
    return;
  }
  release();
  n_ = n;
  glGenBuffers(1, &vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER,
               static_cast<GLsizeiptr>(std::max<std::size_t>(n, 1) * 4 *
                                       sizeof(float)),
               nullptr, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&resource_, vbo_,
                                          cudaGraphicsRegisterFlagsWriteDiscard));
  sums_.resize(4);
}

void GlBodyBuffer::fill(const DeviceBodies &b) {
  const int n = static_cast<int>(b.n);
  if (n == 0 || !resource_) {
    return;
  }
  CUDA_CHECK(cudaGraphicsMapResources(1, &resource_));
  float4 *vertices = nullptr;
  std::size_t bytes = 0;
  CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
      reinterpret_cast<void **>(&vertices), &bytes, resource_));
  CUDA_CHECK(cudaMemset(sums_.data(), 0, 4 * sizeof(float)));
  fillKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(
      n, b.posX.data(), b.posY.data(), b.velX.data(), b.velY.data(),
      b.mass.data(), b.radius.data(), vertices, sums_.data());
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource_));

  float sums[4];
  CUDA_CHECK(cudaMemcpy(sums, sums_.data(), sizeof(sums),
                        cudaMemcpyDeviceToHost));
  if (sums[2] > 0.0f) {
    stats_.comX = sums[0] / sums[2];
    stats_.comY = sums[1] / sums[2];
  }
  stats_.maxSpeed = sums[3] > 0.0f ? sums[3] : 1.0f;
}
