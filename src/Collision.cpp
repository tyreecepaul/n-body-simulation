#include "Collision.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#if defined(_OPENMP) && defined(__GLIBCXX__)
#include <parallel/algorithm>
#endif

namespace {
struct AABB {
  float minX, maxX, minY, maxY;
  int index;
};

void resolve(std::vector<Body> &bodies, int i, int j) {
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

// Strict order the original sort-and-sweep resolved pairs in: by minX,
// with index breaking exact ties (std::sort left those unspecified).
inline bool before(const AABB &a, const AABB &b) {
  return a.minX < b.minX || (a.minX == b.minX && a.index < b.index);
}

// Overlap test for two boxes where `earlier` sorts before `later`, so
// x-overlap reduces to a single comparison. Must match the sweep's
// original conditions exactly to find the same pair set.
inline bool overlaps(const AABB &earlier, const AABB &later) {
  return earlier.maxX >= later.minX && later.minY <= earlier.maxY &&
         earlier.minY <= later.maxY;
}

inline std::int32_t cellCoord(float v, float invCell) {
  constexpr double limit = double(1 << 29);
  double c = std::floor(static_cast<double>(v) * invCell);
  return static_cast<std::int32_t>(std::clamp(c, -limit, limit));
}

struct GridEntry {
  AABB box;
  std::int32_t cx, cy;
};

// A colliding pair, carrying the before() keys of both boxes inline so
// the final sort doesn't chase indices into `boxes`.
struct Pair {
  float laterMinX;
  int later; // index into boxes / bodies
  float earlierMinX;
  int earlier;
};

// Working buffers reused across collide() calls. Several are multi-MB, and
// allocating them fresh every step means re-faulting their pages every
// step. Contents are fully rewritten each call.
struct Scratch {
  std::vector<AABB> boxes;
  std::vector<float> widths;
  std::vector<std::int32_t> cellX, cellY;
  std::vector<unsigned char> largeFlag;
  std::vector<int> large;
  std::vector<std::uint32_t> bucketOf, bucketStart, fill;
  std::vector<GridEntry> grid;
  std::vector<Pair> pairs;
};
} // namespace

void collide(std::vector<Body> &bodies) {
  const std::size_t n = bodies.size();
  if (n < 2) {
    return;
  }

  // Detection only reads start-of-step positions, so its heavy parts (box
  // and cell computation, pair search, pair sort) run in parallel; the
  // counting sort stays serial (random-write bound, atomics didn't pay),
  // as does resolve(), which mutates bodies order-dependently. Integer
  // min/max reductions and the final sort by unique keys keep results
  // identical at any thread count.
  //
  // Function-local, so collide() must not run concurrently with itself
  // (it never does: each simulation calls it from its serial step()).
  static Scratch scratch;

  std::vector<AABB> &boxes = scratch.boxes;
  std::vector<float> &widths = scratch.widths;
  boxes.resize(n);
  widths.resize(n);
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < n; ++i) {
    const Body &body = bodies[i];
    Vec2 pos = body.pos;
    float radius = body.radius;
    Vec2 min = pos - Vec2::one() * radius;
    Vec2 max = pos + Vec2::one() * radius;
    boxes[i] = {min.x, max.x, min.y, max.y, static_cast<int>(i)};
    widths[i] = std::max(max.x - min.x, max.y - min.y);
  }

  // Broadphase: uniform grid with cell size ~ median box width. Each
  // "small" box (no wider than a cell) is binned by its min corner, so any
  // two overlapping small boxes have min corners in the same or adjacent
  // cells. The few "large" boxes (e.g. the heavy central body) are tested
  // against every box directly instead. This replaces the 1D x-axis sweep,
  // whose active list grows with the number of bodies sharing an x-range.
  std::nth_element(widths.begin(), widths.begin() + n / 2, widths.end());
  // 1% slack so float rounding in box widths and cell coordinates can't
  // push an overlapping median-width box outside the 3x3 neighbourhood.
  float cell = widths[n / 2] * 1.01f;
  if (!(cell > 0.0f)) {
    cell = 1.0f;
  }
  const float invCell = 1.0f / cell;
  auto isLarge = [cell](const AABB &b) {
    return b.maxX - b.minX > cell || b.maxY - b.minY > cell;
  };

  // Bin small boxes by cell with a counting sort. Cells are numbered
  // row-major over the occupied extent and folded into ~2 buckets per box,
  // so a cell's horizontal neighbours are adjacent buckets (cache-friendly)
  // and memory stays O(n) even when bodies are flung far out. Distinct
  // cells can share a bucket, so lookups filter on exact cell coordinates.
  std::vector<std::int32_t> &cellX = scratch.cellX;
  std::vector<std::int32_t> &cellY = scratch.cellY;
  std::vector<unsigned char> &largeFlag = scratch.largeFlag;
  cellX.resize(n);
  cellY.resize(n);
  largeFlag.resize(n);
  std::int32_t minCx = std::numeric_limits<std::int32_t>::max();
  std::int32_t minCy = std::numeric_limits<std::int32_t>::max();
  std::int32_t maxCx = std::numeric_limits<std::int32_t>::lowest();
#pragma omp parallel for schedule(static) \
    reduction(min : minCx, minCy) reduction(max : maxCx)
  for (std::size_t i = 0; i < n; ++i) {
    const AABB &b = boxes[i];
    largeFlag[i] = isLarge(b);
    if (largeFlag[i]) {
      continue;
    }
    cellX[i] = cellCoord(b.minX, invCell);
    cellY[i] = cellCoord(b.minY, invCell);
    minCx = std::min(minCx, cellX[i]);
    minCy = std::min(minCy, cellY[i]);
    maxCx = std::max(maxCx, cellX[i]);
  }

  std::uint32_t buckets = 1;
  while (buckets < 2 * n) {
    buckets <<= 1;
  }
  const std::uint32_t mask = buckets - 1;
  // Room for one extra column on each side, so the x-1 / x+1 neighbours
  // of edge columns don't alias into an adjacent row.
  const std::uint64_t rowStride =
      minCx > maxCx ? 1 : static_cast<std::uint64_t>(maxCx - minCx) + 3;
  auto bucket = [&](std::int32_t cx, std::int32_t cy) {
    return static_cast<std::uint32_t>(
               static_cast<std::uint64_t>(cy - minCy + 1) * rowStride +
               static_cast<std::uint64_t>(cx - minCx + 1)) &
           mask;
  };

  std::vector<int> &large = scratch.large;
  std::vector<std::uint32_t> &bucketOf = scratch.bucketOf;
  std::vector<std::uint32_t> &bucketStart = scratch.bucketStart;
  large.clear();
  bucketOf.resize(n);
  bucketStart.assign(buckets + 1, 0);
  for (std::size_t i = 0; i < n; ++i) {
    if (largeFlag[i]) {
      large.push_back(static_cast<int>(i));
      continue;
    }
    bucketOf[i] = bucket(cellX[i], cellY[i]);
    ++bucketStart[bucketOf[i] + 1];
  }
  for (std::uint32_t k = 0; k < buckets; ++k) {
    bucketStart[k + 1] += bucketStart[k];
  }
  std::vector<GridEntry> &grid = scratch.grid;
  grid.resize(n - large.size());
  {
    std::vector<std::uint32_t> &fill = scratch.fill;
    fill.assign(bucketStart.begin(), bucketStart.end() - 1);
    for (std::size_t i = 0; i < n; ++i) {
      if (!largeFlag[i]) {
        grid[fill[bucketOf[i]]++] = {boxes[i], cellX[i], cellY[i]};
      }
    }
  }

  // Half-stencil: each pair of neighbouring cells is visited from exactly
  // one side (same cell, then +x, then the three cells in row +y), so each
  // unordered pair is found once and oriented with before(). Threads
  // collect pairs locally; discovery order is irrelevant because the sort
  // below fixes the resolution order.
  auto test = [](std::vector<Pair> &out, const AABB &a, const AABB &b) {
    const AABB &earlier = before(a, b) ? a : b;
    const AABB &later = before(a, b) ? b : a;
    if (overlaps(earlier, later)) {
      out.push_back({later.minX, later.index, earlier.minX, earlier.index});
    }
  };
  std::vector<Pair> &pairs = scratch.pairs;
  pairs.clear();
#pragma omp parallel
  {
    std::vector<Pair> local;

#pragma omp for schedule(dynamic, 1024) nowait
    for (std::size_t i = 0; i < grid.size(); ++i) {
      const GridEntry &e = grid[i];
      // Same cell: entries after this one in its bucket.
      const std::uint32_t own = bucket(e.cx, e.cy);
      for (std::uint32_t s = i + 1; s < bucketStart[own + 1]; ++s) {
        const GridEntry &o = grid[s];
        if (o.cx == e.cx && o.cy == e.cy) {
          test(local, e.box, o.box);
        }
      }
      // Same row, x + 1.
      const std::uint32_t right = bucket(e.cx + 1, e.cy);
      for (std::uint32_t s = bucketStart[right]; s < bucketStart[right + 1];
           ++s) {
        const GridEntry &o = grid[s];
        if (o.cx == e.cx + 1 && o.cy == e.cy) {
          test(local, e.box, o.box);
        }
      }
      // Row y + 1, x - 1 .. x + 1: three adjacent buckets.
      const std::uint32_t k0 = bucket(e.cx - 1, e.cy + 1);
      for (std::uint32_t j = 0; j < 3; ++j) {
        const std::uint32_t k = (k0 + j) & mask;
        for (std::uint32_t s = bucketStart[k]; s < bucketStart[k + 1]; ++s) {
          const GridEntry &o = grid[s];
          if (o.cy == e.cy + 1 && o.cx >= e.cx - 1 && o.cx <= e.cx + 1) {
            test(local, e.box, o.box);
          }
        }
      }
    }

    for (int l : large) {
#pragma omp for schedule(static) nowait
      for (std::size_t x = 0; x < n; ++x) {
        const AABB &a = boxes[l];
        const AABB &b = boxes[x];
        // Large-vs-large pairs are visited from both ends; keep them once.
        if (static_cast<int>(x) == l || (largeFlag[x] && before(a, b))) {
          continue;
        }
        test(local, a, b);
      }
    }

#pragma omp critical
    pairs.insert(pairs.end(), local.begin(), local.end());
  }

  // resolve() mutates bodies, so order affects results: replay the
  // original sweep's order (by later box, then earlier box), comparing
  // exactly as before() does. Keys are unique, so the result doesn't
  // depend on the order threads appended pairs in.
  // The pair count grows as bodies clump, so late in a run this sort is
  // a large serial cost; use libstdc++'s parallel sort when available.
  auto pairLess = [](const Pair &p, const Pair &q) {
    if (p.later != q.later) {
      return p.laterMinX < q.laterMinX ||
             (p.laterMinX == q.laterMinX && p.later < q.later);
    }
    return p.earlierMinX < q.earlierMinX ||
           (p.earlierMinX == q.earlierMinX && p.earlier < q.earlier);
  };
#if defined(_OPENMP) && defined(__GLIBCXX__)
  __gnu_parallel::sort(pairs.begin(), pairs.end(), pairLess);
#else
  std::sort(pairs.begin(), pairs.end(), pairLess);
#endif
  for (const Pair &p : pairs) {
    resolve(bodies, p.later, p.earlier);
  }
}
