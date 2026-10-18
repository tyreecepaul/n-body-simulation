#include "NaiveSimulation.h"
#include "Collision.h"
#include "Utils.h"
#include <algorithm>
#include <cmath>

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
  // gravitational contribution of every other body individually

  const std::size_t n = bodies.size();

  // Outer loop: compute total acceleration for one body at a time.
  for (std::size_t i = 0; i < n; ++i) {
    Vec2 acc(0.0f, 0.0f);                 // accumulator for body i's net acceleration
    const Vec2 pos_i = bodies[i].pos;     // cached so we don't re-read it every iteration

    // Inner loop: visit every OTHER body and sum its contribution.
    // This is what makes the algorithm O(N) per body, O(N^2) overall.
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) continue;               // a body doesn't attract itself

      Vec2 d = bodies[j].pos - pos_i;     // vector from body i to body j
      float dSq = d.magSq();              // squared distance between the two bodies

      // Softened inverse-square law: eSq prevents forces blowing up
      // (dividing by ~0) when two bodies get very close together.
      float invSoft = 1.0f / (dSq + eSq);
      float invDist = 1.0f / std::sqrt(dSq + eSq);

      // Newtonian gravity scaled by body j's mass; direction is
      // implicitly encoded in vector d, magnitude in factor.
      float factor = bodies[j].mass * invDist * invSoft;
      acc += d * factor; // accumulate this pair's contribution
    }

    // Store the fully-summed acceleration once all N-1 contributions
    // have been added, avoids repeatedly writing to bodies[i] inside
    // the inner loop.
    bodies[i].acc = acc;
  }
}

void NaiveSimulation::iterate() {
  for (Body &body : bodies) {
    body.update(dt);
  }
}

void NaiveSimulation::collide() { ::collide(bodies); }
