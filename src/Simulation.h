#pragma once
#include "Body.h"
#include "QuadTree.h"
#include <cstddef>
#include <vector>

class Simulation {
public:
  float dt;
  std::size_t frame;
  std::vector<Body> bodies;
  QuadTree quadtree;

  explicit Simulation(std::size_t n);

  void step();
  void attract();
  void iterate();
  void collide();
};
