#pragma once
#include "Vec2.h"
#include <cstdint>

struct Body {
  Vec2 pos;
  Vec2 vel;
  Vec2 acc;
  float mass;
  float radius;
  std::uint32_t id; // NEW — stable identity, since QuadTree::build() now
                     // reorders `Simulation::bodies` in place every frame
                     // (see partition-based build), so array position is
                     // no longer a valid way to track "the same body"
                     // across frames/output.

  Body(const Vec2 &pos, const Vec2 &vel, float mass, float radius,
       std::uint32_t id = 0)
      : pos(pos), vel(vel), acc(0.0f, 0.0f), mass(mass), radius(radius),
        id(id) {}

  void update(float dt) {
    vel += acc * dt;
    pos += vel * dt;
  }
};
