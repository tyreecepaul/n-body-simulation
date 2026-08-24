#include "Simulation.h"
#include "Utils.h"
#include <algorithm>
#include <cmath>

namespace {
struct AABB {
  float minX, maxX, minY, maxY;
  int index;
};
} // namespace

Simulation::Simulation(std::size_t n)
    : dt(0.05f), frame(0), bodies(uniformDisc(n)),
      quadtree(/*theta=*/1.0f, /*epsilon=*/1.0f, /*leafCapacity=*/16) {}

void Simulation::step() {
  iterate();
  collide();
  attract();
  frame += 1;
}

void Simulation::attract() {
  // Replaces the old newContaining() + clear() + per-body insert() loop:
  // build() does the containing-quad computation and the whole tree
  // construction in one call via in-place partitioning of `bodies`.
  //
  // NOTE: build() reorders `bodies` in place (see QuadTree::subdivide).
  // That's harmless here — acceleration() below reads pos/mass by
  // whatever index build() leaves things in, and collide()/iterate()
  // don't depend on body order either. It DOES mean array position no
  // longer identifies "the same body" from one frame to the next; see
  // Body::id and Main.cpp's per-frame sort for why that matters for
  // trajectory output.
  quadtree.build(bodies);
  for (Body &body : bodies) {
    body.acc = quadtree.acceleration(body.pos, bodies);
  }
}

void Simulation::iterate() {
  for (Body &body : bodies) {
    body.update(dt);
  }
}

void Simulation::collide() {
  // Broadphase: sort-and-sweep on the x-axis, in place of Rust's
  // broccoli::Tree::find_colliding_pairs. For each AABB (sorted by
  // min x), sweep a set of "active" boxes whose max x hasn't yet been
  // passed, and test full overlap against them.
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

void Simulation::resolve(int i, int j) {
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
    // Relative velocity is ~0: there's no meaningful time-of-impact to
    // solve for (the branch below would divide by ~0 and produce NaN).
    // Just push the bodies directly apart instead. If they're also
    // exactly coincident in position, pick an arbitrary direction.
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
