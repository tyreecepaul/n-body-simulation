#pragma once
#include "CudaCollision.cuh"
#include "CudaQuadTree.cuh"
#include "DeviceBodies.cuh"

#include <cstddef>

// GPU counterpart of Simulation: same initial conditions (uniformDisc),
// dt, and step order (iterate -> collide -> attract), with all state
// resident on the device. As on the CPU, attract() reorders the bodies
// (here into Morton order), so array position is not a stable identity:
// use bodies.id.
class CudaSimulation {
public:
  explicit CudaSimulation(std::size_t n);

  void step();
  void iterate();
  void collide();
  void attract(); // tree build + Barnes-Hut forces

  float dt = 0.05f;
  std::size_t frame = 0;
  DeviceBodies bodies;
  CudaQuadTree tree;

private:
  CudaCollider collider_;
};
