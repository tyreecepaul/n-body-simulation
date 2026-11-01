#pragma once
#include "Vec2.h"
#include <cstdint>

struct Body {
  Vec2 pos;
  Vec2 vel;
  Vec2 acc;
  float mass;
  float radius;
  // Stable identity: QuadTree::build() reorders the body array in place
  // every step, so array position doesn't identify a body across frames.
  std::uint32_t id;

  Body(const Vec2 &pos, const Vec2 &vel, float mass, float radius,
       std::uint32_t id = 0)
      : pos(pos), vel(vel), acc(0.0f, 0.0f), mass(mass), radius(radius),
        id(id) {}

  void update(float dt) {
    vel += acc * dt;
    pos += vel * dt;
  }
};
