#include "QuadTree.h"
#include "Partition.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>

// ----- Newton-Raphson Optimisation ------

inline float fastInvSqrt(float x)
{
    float xhalf = 0.5f * x;

    uint32_t i;
    std::memcpy(&i, &x, sizeof(float));

    i = 0x5f3759df - (i >> 1);

    std::memcpy(&x, &i, sizeof(float));

    // Newton-Raphson refinement
    x = x * (1.5f - xhalf * x * x);

    return x;
}

// ---------------- Quad ----------------

Quad Quad::newContaining(const std::vector<Body> &bodies) {
  float minX = std::numeric_limits<float>::max();
  float minY = std::numeric_limits<float>::max();
  float maxX = std::numeric_limits<float>::lowest();
  float maxY = std::numeric_limits<float>::lowest();

  // min/max are exact, so the reduction is identical at any thread count.
  const std::size_t n = bodies.size();
#pragma omp parallel for schedule(static) \
    reduction(min : minX, minY) reduction(max : maxX, maxY)
  for (std::size_t i = 0; i < n; ++i) {
    minX = std::min(minX, bodies[i].pos.x);
    minY = std::min(minY, bodies[i].pos.y);
    maxX = std::max(maxX, bodies[i].pos.x);
    maxY = std::max(maxY, bodies[i].pos.y);
  }

  Vec2 center = Vec2(minX + maxX, minY + maxY) * 0.5f;
  float size = std::max(maxX - minX, maxY - minY);

  return Quad{center, size};
}

// ---------------- QuadTree ----------------

void QuadTree::subdivide(int node, int children, std::vector<Body> &bodies,
                         std::size_t start, std::size_t end) {
    const Quad &parentQuad = nodes[node].quad;
    const Vec2 center = parentQuad.center;

    // Partition into bottom/top halves.
    const std::size_t mid = partition(
        bodies, start, end,
        [center](const Body &b) {
            return b.pos.y < center.y;
        });

    // Partition each half into left/right.
    const std::size_t leftBottom = partition(
        bodies, start, mid,
        [center](const Body &b) {
            return b.pos.x < center.x;
        });

    const std::size_t rightTop = partition(
        bodies, mid, end,
        [center](const Body &b) {
            return b.pos.x < center.x;
        });

    nodes[node].children = children;

    Quad quads[4];
    parentQuad.subdivide(quads);

    const int next = nodes[node].next;

    nodes[children]     = Node(children + 1, quads[0], start, leftBottom);
    nodes[children + 1] = Node(children + 2, quads[1], leftBottom, mid);
    nodes[children + 2] = Node(children + 3, quads[2], mid, rightTop);
    nodes[children + 3] = Node(next, quads[3], rightTop, end);
}

void QuadTree::accumulateLeaf(int node, const std::vector<Body> &bodies) {
    const std::size_t s = nodes[node].bodyStart;
    const std::size_t e = nodes[node].bodyEnd;

    // Independent accumulators reduce the dependency chain
    // in pos += ... and mass += ...
    Vec2 pos0(0.0f, 0.0f);
    Vec2 pos1(0.0f, 0.0f);
    Vec2 pos2(0.0f, 0.0f);
    Vec2 pos3(0.0f, 0.0f);

    float mass0 = 0.0f;
    float mass1 = 0.0f;
    float mass2 = 0.0f;
    float mass3 = 0.0f;

    std::size_t i = s;

    // Four-way loop unrolling.
    for (; i + 3 < e; i += 4) {
        const Body &b0 = bodies[i];
        const Body &b1 = bodies[i + 1];
        const Body &b2 = bodies[i + 2];
        const Body &b3 = bodies[i + 3];

        pos0 += b0.pos * b0.mass;
        mass0 += b0.mass;

        pos1 += b1.pos * b1.mass;
        mass1 += b1.mass;

        pos2 += b2.pos * b2.mass;
        mass2 += b2.mass;

        pos3 += b3.pos * b3.mass;
        mass3 += b3.mass;
    }

    // Handle remaining 0–3 bodies.
    for (; i < e; ++i) {
        const Body &b = bodies[i];

        pos0 += b.pos * b.mass;
        mass0 += b.mass;
    }

    // Combine independent accumulators once.
    nodes[node].pos = pos0 + pos1 + pos2 + pos3;
    nodes[node].mass = mass0 + mass1 + mass2 + mass3;
}

void QuadTree::processNode(int node, int children,
                           std::vector<Body> &bodies) {
    if (children != 0) {
        subdivide(node, children, bodies,
                  nodes[node].bodyStart, nodes[node].bodyEnd);
    } else {
        accumulateLeaf(node, bodies);
    }
}

void QuadTree::build(std::vector<Body> &bodies) {
    nodes.clear();
    parents.clear();

    Quad quad = Quad::newContaining(bodies);
    nodes.emplace_back(0, quad, 0, bodies.size());

    // Level-synchronous breadth-first build. Serially, a breadth-first
    // build appends each subdividing node's 4 children in node order, so
    // the children of level [levelBegin, levelEnd) land at levelEnd + 4*k,
    // where k is the node's rank among that level's subdividing nodes.
    // Precomputing those slots lets a whole level be processed in
    // parallel (each node partitions its own disjoint body range) while
    // producing exactly the serial node layout and parents order.
    //
    // One parallel region spans all levels, so each level costs barriers
    // rather than a thread fork/join (which dominates for small trees).
    std::size_t levelBegin = 0;
    std::size_t levelEnd = 0;
    std::ptrdiff_t levelSize = 0;
    std::ptrdiff_t chunk = 1;

#pragma omp parallel
    {
        while (true) {
            // The calling thread does the level bookkeeping, and also the
            // root level itself: one node gives nothing to share, and on
            // hybrid CPUs a dynamic schedule might hand it to a slow core.
#if _OPENMP >= 202011
#pragma omp masked
#else
#pragma omp master
#endif
            {
                levelBegin = levelEnd;
                levelEnd = nodes.size();
                levelSize = static_cast<std::ptrdiff_t>(levelEnd - levelBegin);

                childSlot.resize(levelSize);
                int children = static_cast<int>(levelEnd);
                for (std::size_t node = levelBegin; node < levelEnd; ++node) {
                    const Node &n = nodes[node];
                    if (n.bodyEnd - n.bodyStart > leafCapacity) {
                        parents.push_back(static_cast<int>(node));
                        childSlot[node - levelBegin] = children;
                        children += 4;
                    } else {
                        childSlot[node - levelBegin] = 0;
                    }
                }
                nodes.resize(children);

                // Top levels have a handful of nodes, each partitioning a
                // large body range, so they need chunk size 1; deep levels
                // have thousands of small nodes, where larger chunks avoid
                // scheduling contention and false sharing on adjacent Nodes.
                chunk = std::max<std::ptrdiff_t>(1, levelSize / 256);
                if (levelSize == 1) {
                    processNode(static_cast<int>(levelBegin), childSlot[0],
                                bodies);
                }
            }
#pragma omp barrier

            if (levelSize == 0) {
                break;
            }
            if (levelSize > 1) {
#pragma omp for schedule(dynamic, chunk) nowait
                for (std::ptrdiff_t k = 0; k < levelSize; ++k) {
                    processNode(static_cast<int>(levelBegin + k), childSlot[k],
                                bodies);
                }
            }
            // Everyone must be done with this level (and have read
            // levelSize) before the next level's bookkeeping starts.
#pragma omp barrier
        }
    }

    propagate();
}

void QuadTree::propagate() {
    // First: calculate weighted position and mass for every branch.
    for (auto it = parents.rbegin(); it != parents.rend(); ++it) {
        int node = *it;
        int i = nodes[node].children;

        nodes[node].pos =
            nodes[i].pos +
            nodes[i + 1].pos +
            nodes[i + 2].pos +
            nodes[i + 3].pos;

        nodes[node].mass =
            nodes[i].mass +
            nodes[i + 1].mass +
            nodes[i + 2].mass +
            nodes[i + 3].mass;
    }

    // Second: convert weighted positions into centres of mass.
    for (Node &node : nodes) {
        node.pos /=
            std::max(node.mass,
                     std::numeric_limits<float>::min());
    }
}

Vec2 QuadTree::acceleration(const Vec2 &pos,
                            const std::vector<Body> &bodies) const {
    Vec2 acc(0.0f, 0.0f);

    int node = ROOT;

    while (true) {
        const Node &n = nodes[node];

        Vec2 d = n.pos - pos;
        float dSq = d.magSq();

        if (n.quad.size * n.quad.size < dSq * tSq) {

            // Far enough away: use centre-of-mass approximation.
            if (n.mass > 0.0f) {
                float invSqrt = fastInvSqrt(dSq);
                float invSoft = 1.0f / (dSq + eSq);

                float factor = std::min(
                    n.mass * invSqrt * invSoft,
                    std::numeric_limits<float>::max());

                acc += d * factor;
            }

            if (n.next == 0) {
                break;
            }

            node = n.next;

        } else if (n.isLeaf()) {

            // Close leaf: calculate force from individual bodies.
            Vec2 acc0(0.0f, 0.0f);
            Vec2 acc1(0.0f, 0.0f);
            Vec2 acc2(0.0f, 0.0f);
            Vec2 acc3(0.0f, 0.0f);

            std::size_t i = n.bodyStart;
            std::size_t end = n.bodyEnd;

            // Four-way loop unrolling.
            for (; i + 3 < end; i += 4) {

                Vec2 bd0 = bodies[i].pos - pos;
                float bdSq0 = bd0.magSq();
                float invSqrt0 = fastInvSqrt(bdSq0);
                float invSoft0 = 1.0f / (bdSq0 + eSq);

                float factor0 = std::min(
                    bodies[i].mass * invSqrt0 * invSoft0,
                    std::numeric_limits<float>::max());

                acc0 += bd0 * factor0;


                Vec2 bd1 = bodies[i + 1].pos - pos;
                float bdSq1 = bd1.magSq();
                float invSqrt1 = fastInvSqrt(bdSq1);
                float invSoft1 = 1.0f / (bdSq1 + eSq);

                float factor1 = std::min(
                    bodies[i + 1].mass * invSqrt1 * invSoft1,
                    std::numeric_limits<float>::max());

                acc1 += bd1 * factor1;


                Vec2 bd2 = bodies[i + 2].pos - pos;
                float bdSq2 = bd2.magSq();
                float invSqrt2 = fastInvSqrt(bdSq2);
                float invSoft2 = 1.0f / (bdSq2 + eSq);

                float factor2 = std::min(
                    bodies[i + 2].mass * invSqrt2 * invSoft2,
                    std::numeric_limits<float>::max());

                acc2 += bd2 * factor2;


                Vec2 bd3 = bodies[i + 3].pos - pos;
                float bdSq3 = bd3.magSq();
                float invSqrt3 = fastInvSqrt(bdSq3);
                float invSoft3 = 1.0f / (bdSq3 + eSq);

                float factor3 = std::min(
                    bodies[i + 3].mass * invSqrt3 * invSoft3,
                    std::numeric_limits<float>::max());

                acc3 += bd3 * factor3;
            }

            // Remaining 0–3 bodies.
            for (; i < end; ++i) {
                Vec2 bd = bodies[i].pos - pos;
                float bdSq = bd.magSq();
                float invSqrt = fastInvSqrt(bdSq);
                float invSoft = 1.0f / (bdSq + eSq);

                float factor = std::min(
                    bodies[i].mass * invSqrt * invSoft,
                    std::numeric_limits<float>::max());

                acc0 += bd * factor;
            }

            // Combine the four independent accumulators.
            acc += acc0 + acc1 + acc2 + acc3;

            if (n.next == 0) {
                break;
            }

            node = n.next;

        } else {

            // Branch node: descend into its children.
            node = n.children;
        }
    }

    return acc;
}