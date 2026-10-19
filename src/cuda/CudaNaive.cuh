#pragma once
#include "CudaCollision.cuh"
#include "DeviceBodies.cuh"

#include <cstddef>

// Exact O(N^2) gravitational accelerations into b.accX/b.accY: the GPU
// counterpart of NaiveSimulation::attract(), with the same softened
// formula (eSq = softening epsilon squared). Each body sums every other
// body in index order, so results are deterministic run to run.
void naiveAttract(DeviceBodies &b, float eSq);

// GPU counterpart of NaiveSimulation: same initial conditions
// (uniformDisc), dt, softening and step order (iterate -> collide ->
// attract), with all state resident on the device.
class CudaNaiveSimulation {
public:
  explicit CudaNaiveSimulation(std::size_t n);

  void step();

  float dt = 0.05f;
  float eSq = 1.0f * 1.0f; // matches NaiveSimulation's softening
  std::size_t frame = 0;
  DeviceBodies bodies;

private:
  CudaCollider collider_;
};
