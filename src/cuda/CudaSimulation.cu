#include "CudaSimulation.cuh"
#include "Utils.h"

namespace {
constexpr int kBlock = 256;

DeviceBodies uploaded(const std::vector<Body> &initial) {
  DeviceBodies b;
  b.upload(initial);
  return b;
}

// Adds a softened point-mass pull to every body's acceleration, with the
// same form as the tree's force law.
__global__ void attractorKernel(int n, Attractor a,
                                const float *__restrict__ posX,
                                const float *__restrict__ posY,
                                float *__restrict__ accX,
                                float *__restrict__ accY) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float dx = a.x - posX[i];
  const float dy = a.y - posY[i];
  const float dSq = dx * dx + dy * dy;
  const float factor =
      a.mass * rsqrtf(dSq + 1e-12f) / (dSq + a.softening * a.softening);
  accX[i] += dx * factor;
  accY[i] += dy * factor;
}
} // namespace

CudaSimulation::CudaSimulation(std::size_t n)
    : CudaSimulation(uniformDisc(n)) {}

CudaSimulation::CudaSimulation(const std::vector<Body> &initial)
    : bodies(uploaded(initial)), tree(initial.size()), collider_(bodies) {}

void CudaSimulation::reset(const std::vector<Body> &initial) {
  // Release the old buffers before allocating new ones, so peak memory
  // stays at one simulation's worth.
  collider_ = CudaCollider(DeviceBodies{});
  tree = CudaQuadTree(0);
  bodies.upload(initial);
  tree = CudaQuadTree(initial.size());
  collider_ = CudaCollider(bodies);
  frame = 0;
}

void CudaSimulation::step() {
  iterate();
  if (collisions) {
    collide();
  }
  attract();
  frame += 1;
}

void CudaSimulation::iterate() { ::iterate(bodies, dt); }

void CudaSimulation::collide() { collider_.collide(bodies); }

void CudaSimulation::attract() {
  tree.build(bodies);
  tree.computeForces(bodies, theta, epsilon);
  const int n = static_cast<int>(bodies.n);
  if (attractor.active && n > 0) {
    attractorKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(
        n, attractor, bodies.posX.data(), bodies.posY.data(),
        bodies.accX.data(), bodies.accY.data());
    CUDA_CHECK_LAUNCH();
  }
}
