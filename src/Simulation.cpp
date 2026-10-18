#include "Simulation.h"
#include "Collision.h"
#include "Utils.h"
#include <algorithm>
#include <cmath>

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

void Simulation::collide() { ::collide(bodies); }
