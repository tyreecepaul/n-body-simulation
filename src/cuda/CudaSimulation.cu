#include "CudaSimulation.cuh"
#include "Utils.h"

namespace {
DeviceBodies uploaded(std::size_t n) {
  DeviceBodies b;
  b.upload(uniformDisc(n));
  return b;
}
} // namespace

CudaSimulation::CudaSimulation(std::size_t n)
    : bodies(uploaded(n)), tree(n), collider_(bodies) {}

void CudaSimulation::step() {
  iterate();
  collide();
  attract();
  frame += 1;
}

void CudaSimulation::iterate() { ::iterate(bodies, dt); }

void CudaSimulation::collide() { collider_.collide(bodies); }

void CudaSimulation::attract() {
  tree.build(bodies);
  tree.computeForces(bodies);
}
