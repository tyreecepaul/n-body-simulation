// cuda_check: validates the CUDA kernels against the CPU reference code
// and prints PASS/FAIL per test (exit status 1 if anything fails). The GPU
// can't match the CPU bit-for-bit (different float contraction/rounding),
// so each test compares within a stated tolerance.

#include "Collision.h"
#include "CudaCollision.cuh"
#include "CudaNaive.cuh"
#include "CudaQuadTree.cuh"
#include "CudaSimulation.cuh"
#include "DeviceBodies.cuh"
#include "QuadTree.h"
#include "Simulation.h"
#include "Utils.h"
#include "naive/NaiveSimulation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
int failures = 0;

void report(const char *name, bool pass, const char *detailFmt, double a,
            double b) {
  std::printf("[%s] %s: ", pass ? "PASS" : "FAIL", name);
  std::printf(detailFmt, a, b);
  std::printf("\n");
  if (!pass) {
    ++failures;
  }
}

// Per-element relative error |x - ref| / max(|ref|, floor), summarised as
// (max, median).
struct ErrorStats {
  double max = 0.0;
  double median = 0.0;
};

ErrorStats relativeErrors(const std::vector<double> &errors) {
  ErrorStats s;
  if (errors.empty()) {
    return s;
  }
  std::vector<double> sorted = errors;
  std::sort(sorted.begin(), sorted.end());
  s.max = sorted.back();
  s.median = sorted[sorted.size() / 2];
  return s;
}

double vecRelError(Vec2 got, Vec2 ref) {
  const double dx = double(got.x) - ref.x;
  const double dy = double(got.y) - ref.y;
  const double refMag = std::sqrt(double(ref.x) * ref.x + double(ref.y) * ref.y);
  return std::sqrt(dx * dx + dy * dy) / std::max(refMag, 1e-30);
}

void printDevice() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("Device %d: %s (compute %d.%d, %d SMs, %.1f GiB)\n", device,
              prop.name, prop.major, prop.minor, prop.multiProcessorCount,
              prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

// GPU iterate() against Body::update() on the same state.
void testIterate() {
  const std::size_t n = 10000;
  const float dt = 0.05f;
  std::vector<Body> cpu = uniformDisc(n);
  for (std::size_t i = 0; i < n; ++i) {
    // Arbitrary but non-trivial accelerations.
    cpu[i].acc = Vec2(std::sin(0.1f * i), std::cos(0.07f * i));
  }

  DeviceBodies dev;
  dev.upload(cpu);
  iterate(dev, dt);
  std::vector<Body> gpu;
  dev.download(gpu);

  std::vector<double> errors;
  for (std::size_t i = 0; i < n; ++i) {
    cpu[i].update(dt);
    errors.push_back(vecRelError(gpu[i].pos, cpu[i].pos));
    errors.push_back(vecRelError(gpu[i].vel, cpu[i].vel));
  }
  const ErrorStats s = relativeErrors(errors);
  report("iterate vs Body::update", s.max < 1e-5,
         "max rel err %.3g, median %.3g", s.max, s.median);
}

// GPU naiveAttract() against NaiveSimulation::attract() on uniformDisc.
void testNaiveForces() {
  const std::size_t n = 4096;
  NaiveSimulation cpu(n);
  cpu.attract();

  DeviceBodies dev;
  dev.upload(cpu.bodies);
  naiveAttract(dev, /*eSq=*/1.0f);
  std::vector<Body> gpu;
  dev.download(gpu);

  std::vector<double> errors;
  for (std::size_t i = 0; i < n; ++i) {
    errors.push_back(vecRelError(gpu[i].acc, cpu.bodies[i].acc));
  }
  const ErrorStats s = relativeErrors(errors);
  // Sums of ~4k float terms in a different association than the CPU's
  // vectorised loop: expect ~1e-6 typically, allow 1e-3 worst case.
  report("naive forces vs NaiveSimulation::attract (n=4096)",
         s.max < 1e-3 && s.median < 1e-5, "max rel err %.3g, median %.3g",
         s.max, s.median);
}

// Start-of-collide contacts on the CPU, by brute-force sort-and-sweep over
// the same boxes/orientation Collision.cpp uses: the number of overlapping
// box pairs; per body how many boxes it overlaps (and one such partner);
// and per body how many circles it touches.
struct CpuContacts {
  std::int64_t aabbPairs = 0;
  std::vector<int> boxPairs;
  std::vector<int> boxPartner;
  std::vector<int> contacts;
};

CpuContacts cpuContacts(const std::vector<Body> &bodies) {
  struct Box {
    float minX, maxX, minY, maxY;
    int index;
  };
  const std::size_t n = bodies.size();
  std::vector<Box> boxes(n);
  for (std::size_t i = 0; i < n; ++i) {
    const Vec2 min = bodies[i].pos - Vec2::one() * bodies[i].radius;
    const Vec2 max = bodies[i].pos + Vec2::one() * bodies[i].radius;
    boxes[i] = {min.x, max.x, min.y, max.y, static_cast<int>(i)};
  }
  std::sort(boxes.begin(), boxes.end(), [](const Box &a, const Box &b) {
    return a.minX < b.minX || (a.minX == b.minX && a.index < b.index);
  });

  CpuContacts out;
  out.boxPairs.assign(n, 0);
  out.boxPartner.assign(n, -1);
  out.contacts.assign(n, 0);
  for (std::size_t k = 0; k < n; ++k) {
    const Box &later = boxes[k];
    for (std::size_t e = k; e-- > 0;) {
      const Box &earlier = boxes[e];
      if (earlier.maxX < later.minX) {
        // Sorted by minX only, so a short box may hide behind a wide one
        // (the central body): keep scanning but skip non-overlaps.
        continue;
      }
      if (!(later.minY <= earlier.maxY && earlier.minY <= later.maxY)) {
        continue;
      }
      ++out.aabbPairs;
      ++out.boxPairs[later.index];
      ++out.boxPairs[earlier.index];
      out.boxPartner[later.index] = earlier.index;
      out.boxPartner[earlier.index] = later.index;
      const Body &a = bodies[later.index];
      const Body &b = bodies[earlier.index];
      const Vec2 d = b.pos - a.pos;
      const float r = a.radius + b.radius;
      if (d.magSq() > r * r) {
        continue;
      }
      ++out.contacts[later.index];
      ++out.contacts[earlier.index];
    }
  }
  return out;
}

// The CPU Barnes-Hut sim after `steps` steps, advanced to just before its
// next collide(). steps = 0 is the first collide, where bodies at the inner
// edge of the disc still overlap the central body.
std::vector<Body> stateBeforeCollide(std::size_t n, int steps) {
  Simulation sim(n);
  for (int k = 0; k < steps; ++k) {
    sim.step();
  }
  sim.iterate();
  return sim.bodies;
}

// The central body almost never collides in a real run (bodies orbit just
// clear of it), so for the large-body tests place `count` small bodies
// overlapping it, at well-separated angles and with varied velocities.
// Their indices are spread across the array so that, at large n, they fall
// in different chunks of the split scan.
std::vector<Body> withCentralContacts(std::vector<Body> state, int count) {
  std::size_t centre = 0;
  for (std::size_t i = 1; i < state.size(); ++i) {
    if (state[i].radius > state[centre].radius) {
      centre = i;
    }
  }
  const Body c = state[centre];
  for (int k = 0; k < count; ++k) {
    std::size_t i = (k + 1) * state.size() / (count + 1);
    if (i == centre) {
      ++i;
    }
    Body &b = state[i];
    const float angle = 6.2831853f * k / count;
    const Vec2 dir(std::cos(angle), std::sin(angle));
    const float depth = 0.2f + 0.1f * k; // overlap with the central body
    b.pos = c.pos + dir * (c.radius + b.radius - depth);
    // Alternate inward, outward and tangential motion relative to the centre.
    const Vec2 tangent(-dir.y, dir.x);
    const Vec2 dirs[3] = {dir * -5.0f, dir * 3.0f, tangent * 4.0f};
    b.vel = c.vel + dirs[k % 3];
  }
  return state;
}

// A colliding state: after 300 steps the disc has contracted and many
// bodies touch.
std::vector<Body> collidingState(std::size_t n) {
  return stateBeforeCollide(n, 300);
}

bool sameBits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// GPU collide against CPU collide() on the same colliding state.
void testCollision() {
  const std::size_t n = 20000;
  const std::vector<Body> start = collidingState(n);
  const CpuContacts ref = cpuContacts(start);

  std::vector<Body> cpu = start;
  collide(cpu);

  DeviceBodies dev;
  dev.upload(start);
  CudaCollider collider(dev);
  collider.collide(dev);
  std::vector<Body> gpu;
  dev.download(gpu);
  std::vector<int> gpuContacts;
  collider.lastContacts(gpuContacts);

  std::printf("[INFO] collision state: n=%zu, cell=%.4f, %lld box pairs\n", n,
              collider.cellSize(), static_cast<long long>(ref.aabbPairs));

  const std::int64_t gpuPairs = collider.lastAabbPairs();
  report("collide: overlapping box pairs (GPU vs CPU)",
         gpuPairs == ref.aabbPairs, "GPU %.0f, CPU %.0f", double(gpuPairs),
         double(ref.aabbPairs));

  std::size_t contactMismatches = 0;
  std::size_t touching = 0;
  for (std::size_t i = 0; i < n; ++i) {
    contactMismatches += gpuContacts[i] != ref.contacts[i];
    touching += ref.contacts[i] > 0;
  }
  // Borderline circle tests can flip with FMA contraction; allow a handful.
  report("collide: per-body contact counts", contactMismatches <= 2 + n / 10000,
         "%.0f mismatches over %.0f touching bodies", double(contactMismatches),
         double(touching));

  // Bodies with no contacts must come back bit-identical.
  std::size_t untouchedChanged = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (gpuContacts[i] == 0 &&
        !(sameBits(gpu[i].pos.x, start[i].pos.x) &&
          sameBits(gpu[i].pos.y, start[i].pos.y) &&
          sameBits(gpu[i].vel.x, start[i].vel.x) &&
          sameBits(gpu[i].vel.y, start[i].vel.y))) {
      ++untouchedChanged;
    }
  }
  report("collide: bodies without contacts unchanged",
         untouchedChanged == 0, "%.0f changed (of %.0f)",
         double(untouchedChanged), double(n));

  // Isolated pairs must match the CPU's sequential result up to rounding.
  // "Isolated": the two boxes overlap each other and nothing else. (Not just
  // "one circle contact": on the CPU an earlier pair's resolution can push a
  // body into a further contact. Collision.cpp only resolves pairs whose
  // boxes overlapped at the start, so a pair isolated by boxes can't be
  // affected by anything else.)
  std::size_t isolated = 0;
  std::size_t isolatedBad = 0;
  double isolatedMax = 0.0;
  std::vector<double> multiErrors;
  for (std::size_t i = 0; i < n; ++i) {
    if (ref.contacts[i] == 0) {
      continue;
    }
    const auto err = [&](Vec2 got, Vec2 want) {
      const double dx = double(got.x) - want.x;
      const double dy = double(got.y) - want.y;
      const double mag = std::sqrt(double(want.x) * want.x + double(want.y) * want.y);
      return std::sqrt(dx * dx + dy * dy) / std::max(mag, 1.0);
    };
    const double e = std::max(err(gpu[i].pos, cpu[i].pos), err(gpu[i].vel, cpu[i].vel));
    if (ref.contacts[i] == 1 && ref.boxPairs[i] == 1 &&
        ref.boxPairs[ref.boxPartner[i]] == 1) {
      ++isolated;
      isolatedMax = std::max(isolatedMax, e);
      isolatedBad += e > 1e-4;
    } else {
      multiErrors.push_back(e);
    }
  }
  report("collide: isolated pairs match CPU (rel err < 1e-4)",
         isolated > 0 && isolatedBad == 0,
         "%.0f of them differ, max rel err %.3g", double(isolatedBad),
         isolatedMax);
  report("collide: isolated pair count", isolated > 0, "%.0f isolated bodies, %.0f in multi-contacts",
         double(isolated), double(multiErrors.size()));
  if (!multiErrors.empty()) {
    const ErrorStats m = relativeErrors(multiErrors);
    std::printf("[INFO] multi-contact bodies differ from CPU by design "
                "(simultaneous vs sequential): median rel diff %.3g, max %.3g\n",
                m.median, m.max);
  }
}

// The default split large-body scan against the single-block reference:
// small bodies must be bit-identical (their kernel is shared), large ones
// equal up to float summation order.
void testSplitLargeScan(const char *label, const std::vector<Body> &start,
                        int &largeContactsOut) {
  const std::size_t n = start.size();
  std::vector<Body> out[2];
  std::vector<int> contacts[2];
  std::int64_t pairs[2] = {0, 0};
  float cell = 0.0f;
  for (int k = 0; k < 2; ++k) {
    DeviceBodies dev;
    dev.upload(start);
    CudaCollider collider(dev);
    collider.setSplitLargeScan(k == 0);
    collider.collide(dev);
    dev.download(out[k]);
    collider.lastContacts(contacts[k]);
    pairs[k] = collider.lastAabbPairs();
    cell = collider.cellSize();
  }
  // Large = box (2 x radius) wider than a cell, as in the collider.
  std::size_t smallDiffer = 0;
  double largeMax = 0.0;
  int largeBodies = 0;
  int largeContacts = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Body &a = out[0][i];
    const Body &b = out[1][i];
    if (2.0f * start[i].radius > cell) {
      ++largeBodies;
      largeContacts += contacts[0][i];
      const auto err = [](Vec2 g, Vec2 w) {
        const double dx = double(g.x) - w.x, dy = double(g.y) - w.y;
        return std::sqrt(dx * dx + dy * dy) /
               std::max(std::sqrt(double(w.x) * w.x + double(w.y) * w.y), 1.0);
      };
      largeMax = std::max({largeMax, err(a.pos, b.pos), err(a.vel, b.vel)});
    } else {
      smallDiffer += !(sameBits(a.pos.x, b.pos.x) && sameBits(a.pos.y, b.pos.y) &&
                       sameBits(a.vel.x, b.vel.x) && sameBits(a.vel.y, b.vel.y));
    }
  }
  std::printf("[INFO] split large scan (%s): %d large bod%s with %d contacts\n",
              label, largeBodies, largeBodies == 1 ? "y" : "ies", largeContacts);
  largeContactsOut += largeContacts;
  char name[128];
  std::snprintf(name, sizeof name,
                "collide (%s): split large-body scan matches single-block reference",
                label);
  report(name,
         smallDiffer == 0 && contacts[0] == contacts[1] && pairs[0] == pairs[1] &&
             largeMax < 1e-5,
         "%.0f small bodies differ, large max rel diff %.3g",
         double(smallDiffer), largeMax);
}

// Two GPU collides from the same state must give identical bytes.
void testCollisionDeterminism() {
  const std::size_t n = 20000;
  const std::vector<Body> start = collidingState(n);
  std::vector<Body> runs[2];
  for (auto &out : runs) {
    DeviceBodies dev;
    dev.upload(start);
    CudaCollider collider(dev);
    collider.collide(dev);
    dev.download(out);
  }
  std::size_t differing = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Body &a = runs[0][i];
    const Body &b = runs[1][i];
    differing += !(sameBits(a.pos.x, b.pos.x) && sameBits(a.pos.y, b.pos.y) &&
                   sameBits(a.vel.x, b.vel.x) && sameBits(a.vel.y, b.vel.y));
  }
  report("collide: deterministic across runs", differing == 0,
         "%.0f of %.0f bodies differ", double(differing), double(n));
}

// Whole naive simulation, GPU vs CPU, over a few steps (diverges slowly as
// the system is chaotic and rounding differs, so only early steps).
void testNaiveSimulation() {
  const std::size_t n = 2000;
  const int steps = 20;
  NaiveSimulation cpu(n);
  CudaNaiveSimulation gpu(n);
  for (int k = 0; k < steps; ++k) {
    cpu.step();
    gpu.step();
  }
  std::vector<Body> g;
  gpu.bodies.download(g);
  std::vector<double> errors;
  for (std::size_t i = 0; i < n; ++i) {
    errors.push_back(vecRelError(g[i].pos, cpu.bodies[i].pos));
  }
  const ErrorStats s = relativeErrors(errors);
  report("naive simulation, 20 steps (GPU vs CPU positions)",
         s.median < 1e-4 && s.max < 1e-2, "median rel err %.3g, max %.3g",
         s.median, s.max);
}

// Accelerations by body id (bodies may be reordered).
std::vector<Vec2> accById(const std::vector<Body> &bodies) {
  std::vector<Vec2> out(bodies.size());
  for (const Body &b : bodies) {
    out[b.id] = b.acc;
  }
  return out;
}

std::vector<Body> byId(const std::vector<Body> &bodies) {
  std::vector<Body> out(bodies);
  for (const Body &b : bodies) {
    out[b.id] = b;
  }
  return out;
}

// Exact pairwise accelerations in double, with the tree's force law
// (m * d / (|d| * (|d|^2 + eps^2))), so a tree's error against it is the
// Barnes-Hut approximation alone. Input must be in id order.
std::vector<Vec2> exactTreeLawForces(const std::vector<Body> &bodies) {
  const std::size_t n = bodies.size();
  std::vector<Vec2> acc(n);
  for (std::size_t i = 0; i < n; ++i) {
    double ax = 0.0, ay = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      const double dx = double(bodies[j].pos.x) - bodies[i].pos.x;
      const double dy = double(bodies[j].pos.y) - bodies[i].pos.y;
      const double dSq = dx * dx + dy * dy;
      const double f = bodies[j].mass / (std::sqrt(dSq) * (dSq + 1.0));
      ax += dx * f;
      ay += dy * f;
    }
    acc[i] = Vec2(float(ax), float(ay));
  }
  return acc;
}

double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<std::size_t>(p * v.size()))];
}

// Structural checks on a GPU-built tree.
void testTreeInvariants(const char *label, const std::vector<Body> &state) {
  const std::size_t n = state.size();
  DeviceBodies dev;
  dev.upload(state);
  CudaQuadTree tree(n);
  tree.build(dev);
  CudaQuadTree::HostNodes t;
  tree.download(t);
  std::vector<Body> sorted;
  dev.download(sorted);

  double totalMass = 0.0;
  for (const Body &b : state) {
    totalMass += b.mass;
  }
  std::vector<std::pair<int, int>> leaves;
  int oversized = 0;
  int outside = 0;
  for (int node = 0; node < tree.nodeCount(); ++node) {
    if (t.children[node] != 0) {
      continue;
    }
    leaves.emplace_back(t.start[node], t.end[node]);
    oversized += t.end[node] - t.start[node] > 16;
    const float half = t.size[node] * 0.5f * 1.0001f + 1e-3f;
    for (int i = t.start[node]; i < t.end[node]; ++i) {
      outside += std::fabs(sorted[i].pos.x - t.centerX[node]) > half ||
                 std::fabs(sorted[i].pos.y - t.centerY[node]) > half;
    }
  }
  std::sort(leaves.begin(), leaves.end());
  bool tiles = !leaves.empty() && leaves.front().first == 0 &&
               leaves.back().second == static_cast<int>(n);
  for (std::size_t k = 1; k < leaves.size(); ++k) {
    tiles = tiles && leaves[k].first == leaves[k - 1].second;
  }
  // Every id exactly once after the Morton permutation.
  std::vector<int> seen(n, 0);
  bool permutation = true;
  for (const Body &b : sorted) {
    permutation = permutation && b.id < n && seen[b.id]++ == 0;
  }

  std::printf("[INFO] tree (%s): n=%zu, %d nodes (%.3f/body), %d levels\n",
              label, n, tree.nodeCount(), double(tree.nodeCount()) / n,
              tree.levelCount());
  char name[128];
  std::snprintf(name, sizeof name, "tree (%s): root mass = total mass", label);
  report(name, std::fabs(t.mass[0] - totalMass) <= 1e-5 * totalMass,
         "root %.7g, total %.7g", t.mass[0], totalMass);
  std::snprintf(name, sizeof name, "tree (%s): leaves cover every body once, bodies permuted", label);
  report(name, tiles && permutation, "tiles %.0f, permutation %.0f",
         double(tiles), double(permutation));
  std::snprintf(name, sizeof name, "tree (%s): leaves <= 16 bodies, bodies inside their leaf", label);
  report(name, oversized == 0 && outside == 0, "%.0f oversized, %.0f outside",
         double(oversized), double(outside));
}

// The cooperative single-launch build must produce exactly the same tree
// as the host-driven per-level loop.
void testCooperativeBuild(const char *label, const std::vector<Body> &state) {
  const std::size_t n = state.size();
  CudaQuadTree::HostNodes trees[2];
  int nodes[2] = {0, 0};
  int levels[2] = {0, 0};
  bool used[2] = {false, false};
  for (int k = 0; k < 2; ++k) {
    DeviceBodies dev;
    dev.upload(state);
    CudaQuadTree tree(n);
    tree.setUseCooperativeBuild(k == 0);
    used[k] = tree.usesCooperativeBuild();
    tree.build(dev);
    tree.download(trees[k]);
    nodes[k] = tree.nodeCount();
    levels[k] = tree.levelCount();
  }
  const auto sameFloats = [](const std::vector<float> &a, const std::vector<float> &b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
  };
  const CudaQuadTree::HostNodes &a = trees[0];
  const CudaQuadTree::HostNodes &b = trees[1];
  const bool identical =
      nodes[0] == nodes[1] && levels[0] == levels[1] &&
      sameFloats(a.comX, b.comX) && sameFloats(a.comY, b.comY) &&
      sameFloats(a.mass, b.mass) && sameFloats(a.centerX, b.centerX) &&
      sameFloats(a.centerY, b.centerY) && sameFloats(a.size, b.size) &&
      a.children == b.children && a.next == b.next && a.start == b.start &&
      a.end == b.end;
  char name[128];
  std::snprintf(name, sizeof name,
                "tree (%s): cooperative build identical to host loop", label);
  report(name, used[0] && !used[1] && identical,
         "%.0f vs %.0f nodes", double(nodes[0]), double(nodes[1]));
}

// GPU tree forces vs the CPU QuadTree on the same state, and both against
// exact pairwise forces.
void testTreeForces(const char *label, const std::vector<Body> &state) {
  const std::size_t n = state.size();
  DeviceBodies dev;
  dev.upload(state);
  CudaQuadTree tree(n);
  tree.build(dev);
  tree.computeForces(dev);
  std::vector<Body> gpuBodies;
  dev.download(gpuBodies);
  const std::vector<Vec2> gpu = accById(gpuBodies);

  // Same parameters as Simulation's QuadTree.
  std::vector<Body> cpuBodies = state;
  QuadTree cpuTree(/*theta=*/1.0f, /*epsilon=*/1.0f, /*leafCapacity=*/16);
  cpuTree.build(cpuBodies);
  for (Body &b : cpuBodies) {
    b.acc = cpuTree.acceleration(b.pos, cpuBodies);
  }
  const std::vector<Vec2> cpu = accById(cpuBodies);
  const std::vector<Vec2> exact = exactTreeLawForces(byId(state));

  std::vector<double> vsCpu, gpuErr, cpuErr;
  for (std::size_t i = 0; i < n; ++i) {
    vsCpu.push_back(vecRelError(gpu[i], cpu[i]));
    gpuErr.push_back(vecRelError(gpu[i], exact[i]));
    cpuErr.push_back(vecRelError(cpu[i], exact[i]));
  }
  const double vsMedian = percentile(vsCpu, 0.5), vsP99 = percentile(vsCpu, 0.99);
  char name[128];
  std::snprintf(name, sizeof name, "tree forces (%s): GPU vs CPU QuadTree", label);
  // Same tree, different summation order: expect ~1e-7.
  report(name, vsMedian < 1e-5 && vsP99 < 1e-4, "median rel err %.3g, p99 %.3g",
         vsMedian, vsP99);

  const double gMed = percentile(gpuErr, 0.5), gP99 = percentile(gpuErr, 0.99);
  const double cMed = percentile(cpuErr, 0.5), cP99 = percentile(cpuErr, 0.99);
  std::printf("[INFO] tree forces (%s) vs exact: GPU median %.3g p99 %.3g | "
              "CPU median %.3g p99 %.3g\n",
              label, gMed, gP99, cMed, cP99);
  std::snprintf(name, sizeof name, "tree forces (%s): accuracy matches CPU tree", label);
  report(name, gMed <= 2 * cMed + 1e-7 && gP99 <= 2 * cP99 + 1e-7,
         "median ratio %.3g, p99 ratio %.3g", gMed / std::max(cMed, 1e-30),
         gP99 / std::max(cP99, 1e-30));
}

// Whole Barnes-Hut simulation, GPU vs CPU, over a few steps.
void testSimulation() {
  const std::size_t n = 20000;
  const int steps = 20;
  Simulation cpu(n);
  CudaSimulation gpu(n);
  for (int k = 0; k < steps; ++k) {
    cpu.step();
    gpu.step();
  }
  std::vector<Body> g;
  gpu.bodies.download(g);
  const std::vector<Body> gById = byId(g);
  const std::vector<Body> cById = byId(cpu.bodies);
  std::vector<double> errors;
  bool finite = true;
  for (std::size_t i = 0; i < n; ++i) {
    errors.push_back(vecRelError(gById[i].pos, cById[i].pos));
    finite = finite && std::isfinite(gById[i].pos.x) && std::isfinite(gById[i].pos.y);
  }
  const double median = percentile(errors, 0.5);
  report("Barnes-Hut simulation, 20 steps (GPU vs CPU positions)",
         finite && median < 1e-4, "median rel err %.3g, p99 %.3g",
         median, percentile(errors, 0.99));
}

// Same simulation twice on the GPU: identical bytes.
void testSimulationDeterminism() {
  const std::size_t n = 20000;
  std::vector<Body> runs[2];
  for (auto &out : runs) {
    CudaSimulation sim(n);
    for (int k = 0; k < 50; ++k) {
      sim.step();
    }
    sim.bodies.download(out);
  }
  std::size_t differing = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Body &a = runs[0][i];
    const Body &b = runs[1][i];
    differing += a.id != b.id ||
                 !(sameBits(a.pos.x, b.pos.x) && sameBits(a.pos.y, b.pos.y) &&
                   sameBits(a.vel.x, b.vel.x) && sameBits(a.vel.y, b.vel.y));
  }
  report("Barnes-Hut simulation deterministic (50 steps, twice)",
         differing == 0, "%.0f of %.0f bodies differ", double(differing),
         double(n));
}

// Informational: per-phase GPU time per step.
void benchSimulation(std::size_t n, int steps) {
  CudaSimulation sim(n);
  sim.step(); // warm-up
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t ev[5];
  for (auto &e : ev) {
    CUDA_CHECK(cudaEventCreate(&e));
  }
  float phase[4] = {0, 0, 0, 0};
  for (int k = 0; k < steps; ++k) {
    CUDA_CHECK(cudaEventRecord(ev[0]));
    sim.iterate();
    CUDA_CHECK(cudaEventRecord(ev[1]));
    sim.collide();
    CUDA_CHECK(cudaEventRecord(ev[2]));
    sim.tree.build(sim.bodies);
    CUDA_CHECK(cudaEventRecord(ev[3]));
    sim.tree.computeForces(sim.bodies);
    CUDA_CHECK(cudaEventRecord(ev[4]));
    CUDA_CHECK(cudaEventSynchronize(ev[4]));
    for (int p = 0; p < 4; ++p) {
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev[p], ev[p + 1]));
      phase[p] += ms;
    }
  }
  for (auto &e : ev) {
    CUDA_CHECK(cudaEventDestroy(e));
  }
  const float total = phase[0] + phase[1] + phase[2] + phase[3];
  std::printf("[INFO] Barnes-Hut n=%zu ms/step: total %.3f | iterate %.3f "
              "collide %.3f build %.3f forces %.3f (%d nodes)\n",
              n, total / steps, phase[0] / steps, phase[1] / steps,
              phase[2] / steps, phase[3] / steps, sim.tree.nodeCount());
}

// Informational: naive kernel time at a size where the CPU takes seconds.
void benchNaive() {
  const std::size_t n = 100000;
  DeviceBodies dev;
  dev.upload(uniformDisc(n));
  naiveAttract(dev, 1.0f); // warm-up
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  const int reps = 5;
  CUDA_CHECK(cudaEventRecord(start));
  for (int r = 0; r < reps; ++r) {
    naiveAttract(dev, 1.0f);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::printf("[INFO] naive attract n=%zu: %.3f ms per call\n", n, ms / reps);
}
} // namespace

int main() {
  printDevice();
  testIterate();
  testNaiveForces();
  testCollision();
  testCollisionDeterminism();
  {
    // At least one state must exercise large-body contacts for real.
    int largeContacts = 0;
    testSplitLargeScan("after 300 steps", collidingState(20000), largeContacts);
    testSplitLargeScan("8 forced central contacts",
                       withCentralContacts(collidingState(20000), 8),
                       largeContacts);
    testSplitLargeScan("8 forced central contacts, n=1M",
                       withCentralContacts(stateBeforeCollide(1000000, 0), 8),
                       largeContacts);
    report("collide: large-body tests saw real contacts", largeContacts > 0,
           "%.0f contacts across %.0f states", double(largeContacts), 3.0);
  }
  testNaiveSimulation();
  {
    const std::vector<Body> disc = uniformDisc(20000);
    const std::vector<Body> colliding = collidingState(20000);
    testTreeInvariants("initial disc", disc);
    testTreeInvariants("after 300 steps", colliding);
    testCooperativeBuild("initial disc", disc);
    testCooperativeBuild("after 300 steps", colliding);
    testTreeForces("initial disc", disc);
    testTreeForces("after 300 steps", colliding);
  }
  testSimulation();
  testSimulationDeterminism();
  benchNaive();
  benchSimulation(10000, 50);
  benchSimulation(100000, 20);
  benchSimulation(1000000, 5);
  std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASSED" : "FAILED",
              failures, failures == 1 ? "" : "s");
  return failures == 0 ? 0 : 1;
}
