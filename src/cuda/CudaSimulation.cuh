#pragma once
#include "CudaCollision.cuh"
#include "CudaQuadTree.cuh"
#include "DeviceBodies.cuh"
#include <cstddef>
#include <vector>

// A point mass that pulls on every body without being simulated itself
// (the viewer puts it under the mouse cursor).
struct Attractor {
  bool active = false;
  float x = 0.0f, y = 0.0f;
  float mass = 1e6f;
  float softening = 50.0f; // keeps the pull finite near the point
};

// GPU counterpart of Simulation: same initial conditions (uniformDisc),
// dt, and step order (iterate -> collide -> attract), with all state
// resident on the device. As on the CPU, attract() reorders the bodies
// (here into Morton order), so array position is not a stable identity:
// use bodies.id.
//
// The defaults (theta = epsilon = 1, collisions on, no attractor) match
// the CPU simulation; the viewer changes them at run time.
class CudaSimulation {
public:
  explicit CudaSimulation(std::size_t n);
  explicit CudaSimulation(const std::vector<Body> &initial);

  // Replaces every body (any count), reallocating the tree and collision
  // buffers for the new size. Parameters are kept.
  void reset(const std::vector<Body> &initial);

  void step();
  void iterate();
  void collide();
  void attract(); // tree build + Barnes-Hut forces (+ attractor)

  float dt = 0.05f;
  float theta = tree_detail::kTheta;
  float epsilon = tree_detail::kEpsilon;
  bool collisions = true;
  Attractor attractor;

  std::size_t frame = 0;
  DeviceBodies bodies;
  CudaQuadTree tree;

private:
  CudaCollider collider_;
};
