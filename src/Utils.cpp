#include "Utils.h"
#include <algorithm>
#include <cmath>
#include <random>

std::vector<Body> uniformDisc(std::size_t n) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);

  const float innerRadius = 25.0f;
  const float outerRadius = std::sqrt(static_cast<float>(n)) * 5.0f;

  std::vector<Body> bodies;
  bodies.reserve(n);

  const float m = 1e6f;
  Body center(Vec2::zero(), Vec2::zero(), m, innerRadius, /*id=*/0);
  bodies.push_back(center);

  // ids are assigned here, at creation, before the sort-by-radius below
  // and before any QuadTree::build() reordering happens during
  // simulation — so id stays a stable identity for a body no matter how
  // its array position later moves. See Body.h.
  std::uint32_t nextId = 1;
  while (bodies.size() < n) {
    const float a = unit(rng) * static_cast<float>(M_PI) * 2.0f; // TAU
    const float sin = std::sin(a);
    const float cos = std::cos(a);
    const float t = innerRadius / outerRadius;
    const float r = unit(rng) * (1.0f - t * t) + t * t;
    const Vec2 pos = Vec2(cos, sin) * outerRadius * std::sqrt(r);
    const Vec2 vel = Vec2(sin, -cos);
    const float mass = 1.0f;
    const float radius = std::cbrt(mass);

    bodies.emplace_back(pos, vel, mass, radius, nextId++);
  }

  std::sort(bodies.begin(), bodies.end(), [](const Body &a, const Body &b) {
    return a.pos.magSq() < b.pos.magSq();
  });

  float mass = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    mass += bodies[i].mass;
    if (bodies[i].pos == Vec2::zero()) {
      continue;
    }
    float v = std::sqrt(mass / bodies[i].pos.mag());
    bodies[i].vel *= v;
  }

  return bodies;
}
