#pragma once
#include "Body.h"
#include <cstddef>
#include <vector>

// Naive O(N^2) counterpart to Simulation (which uses the Barnes-Hut
// QuadTree). Same public interface, same collision handling, same
// integration scheme, and the same softening epsilon as QuadTree's eSq —
// the only difference is how attract() computes acceleration, so results
// and timings are directly comparable against the Barnes-Hut version.
class NaiveSimulation {
public:
  float dt;
  std::size_t frame;
  std::vector<Body> bodies;

  explicit NaiveSimulation(std::size_t n);

  void step();
  void attract();
  void iterate();
  void collide();

private:
  void resolve(int i, int j);

  float eSq; // softening epsilon squared, matches QuadTree's default (epsilon=1.0f)
};