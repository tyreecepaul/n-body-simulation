#include "CudaNaive.cuh"
#include "Utils.h"

namespace {
// Threads per block, and bodies per shared-memory tile (must be equal:
// each thread loads one body of the tile).
constexpr int kTile = 256;

// Tiled all-pairs kernel: every block walks the body array in tiles staged
// through shared memory, so each position/mass is read from global memory
// once per block instead of once per thread.
//
// Formula per pair, as in NaiveSimulation::attract():
//   d = pos_j - pos_i, s = |d|^2 + eSq, acc_i += d * m_j / (sqrt(s) * s)
// No i == j branch is needed: d = 0 contributes exactly 0 (s >= eSq > 0).
// Padding slots past n get mass 0, which also contributes exactly 0.
__global__ void naiveAttractKernel(int n, float eSq,
                                   const float *__restrict__ posX,
                                   const float *__restrict__ posY,
                                   const float *__restrict__ mass,
                                   float *__restrict__ accX,
                                   float *__restrict__ accY) {
  __shared__ float tileX[kTile];
  __shared__ float tileY[kTile];
  __shared__ float tileM[kTile];

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const float xi = i < n ? posX[i] : 0.0f;
  const float yi = i < n ? posY[i] : 0.0f;
  float ax = 0.0f;
  float ay = 0.0f;

  for (int tile = 0; tile < n; tile += kTile) {
    const int j = tile + threadIdx.x;
    tileX[threadIdx.x] = j < n ? posX[j] : 0.0f;
    tileY[threadIdx.x] = j < n ? posY[j] : 0.0f;
    tileM[threadIdx.x] = j < n ? mass[j] : 0.0f;
    __syncthreads();

#pragma unroll 8
    for (int k = 0; k < kTile; ++k) {
      const float dx = tileX[k] - xi;
      const float dy = tileY[k] - yi;
      const float s = dx * dx + dy * dy + eSq;
      const float invDist = rsqrtf(s);
      const float factor = tileM[k] * invDist * (1.0f / s);
      ax += dx * factor;
      ay += dy * factor;
    }
    __syncthreads();
  }

  if (i < n) {
    accX[i] = ax;
    accY[i] = ay;
  }
}
} // namespace

void naiveAttract(DeviceBodies &b, float eSq) {
  const int n = static_cast<int>(b.n);
  if (n == 0) {
    return;
  }
  naiveAttractKernel<<<(n + kTile - 1) / kTile, kTile>>>(
      n, eSq, b.posX.data(), b.posY.data(), b.mass.data(), b.accX.data(),
      b.accY.data());
  CUDA_CHECK_LAUNCH();
}

namespace {
DeviceBodies uploaded(std::size_t n) {
  DeviceBodies b;
  b.upload(uniformDisc(n));
  return b;
}
} // namespace

CudaNaiveSimulation::CudaNaiveSimulation(std::size_t n)
    : bodies(uploaded(n)), collider_(bodies) {}

void CudaNaiveSimulation::step() {
  iterate(bodies, dt);
  collider_.collide(bodies);
  naiveAttract(bodies, eSq);
  frame += 1;
}
