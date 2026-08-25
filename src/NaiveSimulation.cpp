#include "NaiveSimulation.h"
#include "Utils.h"
#include <algorithm>
#include <cmath>

namespace {
struct AABB {
  float minX, maxX, minY, maxY;
  int index;
};
} // namespace

NaiveSimulation::NaiveSimulation(std::size_t n)
    : dt(0.05f), frame(0), bodies(uniformDisc(n)), eSq(1.0f * 1.0f) {}

void NaiveSimulation::step() {
  iterate();
  collide();
  attract();
  frame += 1;
}

void NaiveSimulation::attract() {
  // Direct O(N^2) pairwise summation: for every body, sum the
  // gravitational contribution of every other body individually.
  // No spatial structure, no approximation — this is the ground-truth
  // reference that Barnes-Hut is checked against, and the baseline that
  // its O(N log N) tree walk is compared to for performance.
  const std::size_t n = bodies.size();
  for (std::size_t i = 0; i < n; ++i) {
    Vec2 acc(0.0f, 0.0f);
    const Vec2 pos_i = bodies[i].pos;

    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) continue;

      Vec2 d = bodies[j].pos - pos_i;
      float dSq = d.magSq();
      float invSoft = 1.0f / (dSq + eSq);
      float invDist = 1.0f / std::sqrt(dSq + eSq);

      float factor = bodies[j].mass * invDist * invSoft;
      acc += d * factor;
    }

    bodies[i].acc = acc;
  }
}

void NaiveSimulation::iterate() {
  for (Body &body : bodies) {
    body.update(dt);
  }
}

void NaiveSimulation::collide() {
  // Same sort-and-sweep broadphase as Simulation::collide(), kept
  // identical so any performance/behaviour difference between the naive
  // and Barnes-Hut simulations comes only from attract(), not collision
  // handling.
  std::vector<AABB> boxes;
  boxes.reserve(bodies.size());
  for (std::size_t i = 0; i < bodies.size(); ++i) {
    const Body &body = bodies[i];
    Vec2 pos = body.pos;
    float radius = body.radius;
    Vec2 min = pos - Vec2::one() * radius;
    Vec2 max = pos + Vec2::one() * radius;
    boxes.push_back({min.x, max.x, min.y, max.y, static_cast<int>(i)});
  }

  std::sort(boxes.begin(), boxes.end(),
            [](const AABB &a, const AABB &b) { return a.minX < b.minX; });

  std::vector<AABB> active;
  for (const AABB &box : boxes) {
    active.erase(
        std::remove_if(active.begin(), active.end(),
                       [&](const AABB &a) { return a.maxX < box.minX; }),
        active.end());

    for (const AABB &other : active) {
      if (box.minY <= other.maxY && other.minY <= box.maxY) {
        resolve(box.index, other.index);
      }
    }

    active.push_back(box);
  }
}

void NaiveSimulation::resolve(int i, int j) {
  const Body &b1 = bodies[i];
  const Body &b2 = bodies[j];
  Vec2 p1 = b1.pos;
  Vec2 p2 = b2.pos;
  float r1 = b1.radius;
  float r2 = b2.radius;

  Vec2 d = p2 - p1;
  float r = r1 + r2;
  if (d.magSq() > r * r) {
    return;
  }

  Vec2 v1 = b1.vel;
  Vec2 v2 = b2.vel;
  Vec2 v = v2 - v1;
  float dDotV = d.dot(v);

  float m1 = b1.mass;
  float m2 = b2.mass;
  float weight1 = m2 / (m1 + m2);
  float weight2 = m1 / (m1 + m2);

  if (dDotV >= 0.0f && d != Vec2::zero()) {
    Vec2 tmp = d * (r / d.mag() - 1.0f);
    bodies[i].pos -= weight1 * tmp;
    bodies[j].pos += weight2 * tmp;
    return;
  }

  float vSqEarly = v.magSq();
  if (vSqEarly < 1e-12f) {
    Vec2 dir = (d != Vec2::zero()) ? d * (1.0f / d.mag()) : Vec2(1.0f, 0.0f);
    Vec2 tmp = dir * r;
    bodies[i].pos -= weight1 * tmp;
    bodies[j].pos += weight2 * tmp;
    return;
  }

  float vSq = v.magSq();
  float dSq = d.magSq();
  float rSq = r * r;
  float t =
      (dDotV + std::sqrt(std::max(dDotV * dDotV - vSq * (dSq - rSq), 0.0f))) /
      vSq;

  bodies[i].pos -= v1 * t;
  bodies[j].pos -= v2 * t;

  p1 = bodies[i].pos;
  p2 = bodies[j].pos;
  d = p2 - p1;
  dDotV = d.dot(v);
  dSq = d.magSq();
  Vec2 tmp = d * (1.5f * dDotV / dSq);

  Vec2 nv1 = v1 + tmp * weight1;
  Vec2 nv2 = v2 - tmp * weight2;

  bodies[i].vel = nv1;
  bodies[j].vel = nv2;
  bodies[i].pos += nv1 * t;
  bodies[j].pos += nv2 * t;
}
