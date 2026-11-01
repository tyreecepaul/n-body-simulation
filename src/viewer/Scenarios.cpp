#include "Scenarios.h"
#include "Utils.h"

#include <cmath>
#include <random>

namespace {
// A uniformDisc moved to (x, y) with bulk velocity (vx, vy); `reverse`
// flips its rotation.
void appendDisc(std::vector<Body> &out, std::size_t n, unsigned seed,
                float x, float y, float vx, float vy, bool reverse) {
  for (Body b : uniformDisc(n, seed)) {
    if (reverse) {
      b.vel = b.vel * -1.0f;
    }
    b.pos += Vec2(x, y);
    b.vel += Vec2(vx, vy);
    out.push_back(b);
  }
}
} // namespace

void renumber(std::vector<Body> &bodies) {
  for (std::size_t i = 0; i < bodies.size(); ++i) {
    bodies[i].id = static_cast<std::uint32_t>(i);
  }
}

std::vector<Body> makeScenario(Scenario s, std::size_t n, unsigned seed) {
  std::vector<Body> bodies;
  bodies.reserve(n);
  switch (s) {
  case Scenario::Disc:
    bodies = uniformDisc(n, seed);
    break;
  case Scenario::TwoGalaxies: {
    // Two discs on a glancing collision course, rotating opposite ways.
    // uniformDisc's radius is 5 * sqrt(n); the approach speed is a
    // fraction of the circular speed at the discs' separation.
    const std::size_t half = n / 2;
    const float radius = 5.0f * std::sqrt(static_cast<float>(half));
    const float sep = 2.5f * radius;
    const float totalMass = 2.0f * (1e6f + static_cast<float>(half));
    const float v = 0.3f * std::sqrt(totalMass / sep);
    appendDisc(bodies, half, seed, -sep / 2, -radius / 3, v, 0.0f, false);
    appendDisc(bodies, n - half, seed + 1, sep / 2, radius / 3, -v, 0.0f,
               true);
    break;
  }
  case Scenario::ColdSquare: {
    std::mt19937 rng(seed);
    const float half = 3.0f * std::sqrt(static_cast<float>(n));
    std::uniform_real_distribution<float> coord(-half, half);
    for (std::size_t i = 0; i < n; ++i) {
      bodies.emplace_back(Vec2(coord(rng), coord(rng)), Vec2(), 1.0f, 1.0f);
    }
    break;
  }
  }
  renumber(bodies);
  return bodies;
}

void addHeavyBody(std::vector<Body> &bodies, float x, float y, float mass) {
  // Same density as the central body: radius grows with the cube root.
  const float radius = 25.0f * std::cbrt(mass / 1e6f);
  bodies.emplace_back(Vec2(x, y), Vec2(), mass, std::fmax(radius, 1.0f),
                      static_cast<std::uint32_t>(bodies.size()));
}

void addCluster(std::vector<Body> &bodies, float x, float y, float vx,
                float vy, std::size_t count, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  const float spread = 2.0f * std::sqrt(static_cast<float>(count));
  for (std::size_t i = 0; i < count; ++i) {
    const float a = unit(rng) * 6.2831853f;
    const float r = spread * std::sqrt(unit(rng));
    bodies.emplace_back(Vec2(x + r * std::cos(a), y + r * std::sin(a)),
                        Vec2(vx, vy), 1.0f, 1.0f,
                        static_cast<std::uint32_t>(bodies.size()));
  }
}
