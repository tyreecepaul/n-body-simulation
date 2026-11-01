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

  // Each body only reads the (const) tree and writes its own acc, so the
  // loop is race-free and bit-identical to serial at any thread count.
  // Dynamic schedule because bodies near the dense centre open far more
  // nodes; build() leaves bodies spatially ordered, so a chunk of
  // consecutive bodies walks a similar part of the tree.
  const std::size_t n = bodies.size();
#pragma omp parallel for schedule(dynamic, 128)
  for (std::size_t i = 0; i < n; ++i) {
    bodies[i].acc = quadtree.acceleration(bodies[i].pos, bodies);
  }
}

void Simulation::iterate() {
  const std::size_t n = bodies.size();
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < n; ++i) {
    bodies[i].update(dt);
  }
}

void Simulation::collide() { ::collide(bodies); }
