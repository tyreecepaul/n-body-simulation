#pragma once
// Per-body collision logic used by CudaCollision.cu's kernels. Written as
// plain __host__ __device__ functions so the kernels stay thin wrappers and
// the same code can also be compiled and run on the host (e.g. to check
// the algorithm against Collision.cpp without a GPU).

#include <math.h>

#ifdef __CUDACC__
#define NBODY_HD __host__ __device__
#else
#define NBODY_HD
#endif

namespace collision_detail {

constexpr unsigned long long kLargeKey = ~0ull; // large boxes sort last

struct Box {
  float minX, maxX, minY, maxY;
  int index;
};

// Same box as Collision.cpp: pos -/+ radius on each axis.
NBODY_HD inline Box makeBox(int i, const float *posX, const float *posY,
                            const float *radius) {
  const float x = posX[i];
  const float y = posY[i];
  const float r = radius[i];
  return Box{x - r, x + r, y - r, y + r, i};
}

// Strict order of Collision.cpp's before(): by minX, index breaks ties.
NBODY_HD inline bool before(const Box &a, const Box &b) {
  return a.minX < b.minX || (a.minX == b.minX && a.index < b.index);
}

// Collision.cpp's overlaps(), for `earlier` sorting before `later`.
NBODY_HD inline bool overlaps(const Box &earlier, const Box &later) {
  return earlier.maxX >= later.minX && later.minY <= earlier.maxY &&
         earlier.minY <= later.maxY;
}

NBODY_HD inline bool isLarge(const Box &b, float cell) {
  return b.maxX - b.minX > cell || b.maxY - b.minY > cell;
}

NBODY_HD inline int cellCoord(float v, float invCell) {
  // Clamp (also maps NaN to a finite value) before converting to int.
  const float c = fminf(fmaxf(floorf(v * invCell), -536870912.0f),
                        536870912.0f); // +-2^29
  return static_cast<int>(c);
}

// Row-major sortable key: within a row (fixed cy), keys are contiguous in
// cx, so the 3 cells cx-1..cx+1 form one key range.
NBODY_HD inline unsigned long long cellKey(int cx, int cy) {
  const long long bias = 1ll << 30;
  return (static_cast<unsigned long long>(cy + bias) << 32) |
         static_cast<unsigned long long>(cx + bias);
}

// Grid key of a body's box: kLargeKey for boxes wider than a cell.
NBODY_HD inline unsigned long long boxKey(const Box &box, float cell,
                                          float invCell) {
  if (isLarge(box, cell)) {
    return kLargeKey;
  }
  return cellKey(cellCoord(box.minX, invCell), cellCoord(box.minY, invCell));
}

NBODY_HD inline int lowerBound(const unsigned long long *keys, int count,
                               unsigned long long key) {
  int lo = 0;
  int hi = count;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (keys[mid] < key) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Read-only view of the start-of-collide state.
struct BodyArrays {
  const float *posX, *posY, *velX, *velY, *mass, *radius;
};

struct BodyState {
  float px, py, vx, vy, m, r;
};

NBODY_HD inline BodyState loadState(int i, const BodyArrays &b) {
  return BodyState{b.posX[i], b.posY[i], b.velX[i],
                   b.velY[i], b.mass[i], b.radius[i]};
}

// Port of Collision.cpp's resolve(bodies, i, j) for i = s1 (the pair's
// later box) and j = s2, returning both bodies' new pos/vel instead of
// writing them. Returns false (no change) if the circles don't touch.
NBODY_HD inline bool resolvePair(const BodyState &s1, const BodyState &s2,
                                 float &p1x, float &p1y, float &v1x,
                                 float &v1y, float &p2x, float &p2y,
                                 float &v2x, float &v2y) {
  float dx = s2.px - s1.px;
  float dy = s2.py - s1.py;
  const float r = s1.r + s2.r;
  if (dx * dx + dy * dy > r * r) {
    return false;
  }

  const float vx = s2.vx - s1.vx;
  const float vy = s2.vy - s1.vy;
  float dDotV = dx * vx + dy * vy;

  const float weight1 = s2.m / (s1.m + s2.m);
  const float weight2 = s1.m / (s1.m + s2.m);
  const bool dNonZero = dx != 0.0f || dy != 0.0f;

  v1x = s1.vx;
  v1y = s1.vy;
  v2x = s2.vx;
  v2y = s2.vy;

  if (dDotV >= 0.0f && dNonZero) {
    // Separating: push apart to touching distance.
    const float s = r / sqrtf(dx * dx + dy * dy) - 1.0f;
    const float tx = dx * s;
    const float ty = dy * s;
    p1x = s1.px - weight1 * tx;
    p1y = s1.py - weight1 * ty;
    p2x = s2.px + weight2 * tx;
    p2y = s2.py + weight2 * ty;
    return true;
  }

  const float vSq = vx * vx + vy * vy;
  if (vSq < 1e-12f) {
    // ~zero relative velocity: no time of impact; push directly apart.
    float dirX = 1.0f;
    float dirY = 0.0f;
    if (dNonZero) {
      const float inv = 1.0f / sqrtf(dx * dx + dy * dy);
      dirX = dx * inv;
      dirY = dy * inv;
    }
    const float tx = dirX * r;
    const float ty = dirY * r;
    p1x = s1.px - weight1 * tx;
    p1y = s1.py - weight1 * ty;
    p2x = s2.px + weight2 * tx;
    p2y = s2.py + weight2 * ty;
    return true;
  }

  // Rewind to time of impact, exchange momentum, replay the same time.
  const float dSq = dx * dx + dy * dy;
  const float rSq = r * r;
  const float t =
      (dDotV + sqrtf(fmaxf(dDotV * dDotV - vSq * (dSq - rSq), 0.0f))) / vSq;

  const float q1x = s1.px - s1.vx * t;
  const float q1y = s1.py - s1.vy * t;
  const float q2x = s2.px - s2.vx * t;
  const float q2y = s2.py - s2.vy * t;

  dx = q2x - q1x;
  dy = q2y - q1y;
  dDotV = dx * vx + dy * vy;
  const float s = 1.5f * dDotV / (dx * dx + dy * dy);
  const float tx = dx * s;
  const float ty = dy * s;

  v1x = s1.vx + tx * weight1;
  v1y = s1.vy + ty * weight1;
  v2x = s2.vx - tx * weight2;
  v2y = s2.vy - ty * weight2;
  p1x = q1x + v1x * t;
  p1y = q1y + v1y * t;
  p2x = q2x + v2x * t;
  p2y = q2y + v2y * t;
  return true;
}

// One body's view of its contacts.
struct Accum {
  float dpx = 0.0f, dpy = 0.0f, dvx = 0.0f, dvy = 0.0f; // summed changes
  float npx = 0.0f, npy = 0.0f, nvx = 0.0f, nvy = 0.0f; // summed new state
  int contacts = 0;
  int aabbAsLater = 0; // box pairs where this body is the later box
};

// Tests body a against body b and adds a's side of the pair to acc.
NBODY_HD inline void accumulatePair(const Box &boxA, const BodyState &sa,
                                    int b, const BodyArrays &bodies,
                                    Accum &acc) {
  const Box boxB = makeBox(b, bodies.posX, bodies.posY, bodies.radius);
  const bool aIsLater = before(boxB, boxA);
  if (!(aIsLater ? overlaps(boxB, boxA) : overlaps(boxA, boxB))) {
    return;
  }
  if (aIsLater) {
    ++acc.aabbAsLater;
  }

  const BodyState sb = loadState(b, bodies);
  float p1x, p1y, v1x, v1y, p2x, p2y, v2x, v2y;
  const bool hit =
      aIsLater
          ? resolvePair(sa, sb, p1x, p1y, v1x, v1y, p2x, p2y, v2x, v2y)
          : resolvePair(sb, sa, p1x, p1y, v1x, v1y, p2x, p2y, v2x, v2y);
  if (!hit) {
    return;
  }
  // a is body 1 of the pair if it's the later box, body 2 otherwise.
  const float px = aIsLater ? p1x : p2x;
  const float py = aIsLater ? p1y : p2y;
  const float vx = aIsLater ? v1x : v2x;
  const float vy = aIsLater ? v1y : v2y;
  ++acc.contacts;
  acc.dpx += px - sa.px;
  acc.dpy += py - sa.py;
  acc.dvx += vx - sa.vx;
  acc.dvy += vy - sa.vy;
  acc.npx += px;
  acc.npy += py;
  acc.nvx += vx;
  acc.nvy += vy;
}

// All contacts of small body indexSorted[s]: the full 3x3 stencil of
// small boxes (sorted by cell key in [0, nSmall)) plus every large box
// ([nSmall, n)), scanned in a fixed order.
NBODY_HD inline Accum smallBodyContacts(int s, int n, int nSmall,
                                        float invCell,
                                        const unsigned long long *keysSorted,
                                        const int *indexSorted,
                                        const BodyArrays &bodies) {
  const int a = indexSorted[s];
  const Box boxA = makeBox(a, bodies.posX, bodies.posY, bodies.radius);
  const BodyState sa = loadState(a, bodies);
  const int cx = cellCoord(boxA.minX, invCell);
  const int cy = cellCoord(boxA.minY, invCell);

  Accum acc;
  for (int dy = -1; dy <= 1; ++dy) {
    const unsigned long long lastKey = cellKey(cx + 1, cy + dy);
    for (int e = lowerBound(keysSorted, nSmall, cellKey(cx - 1, cy + dy));
         e < nSmall && keysSorted[e] <= lastKey; ++e) {
      const int b = indexSorted[e];
      if (b != a) {
        accumulatePair(boxA, sa, b, bodies, acc);
      }
    }
  }
  for (int e = nSmall; e < n; ++e) {
    accumulatePair(boxA, sa, indexSorted[e], bodies, acc);
  }
  return acc;
}

// Final state of body a from its accumulated contacts.
NBODY_HD inline void writeResult(int a, const BodyState &sa, const Accum &acc,
                                 float *outPosX, float *outPosY,
                                 float *outVelX, float *outVelY,
                                 int *contacts, int *aabbAsLater) {
  float px = sa.px, py = sa.py, vx = sa.vx, vy = sa.vy;
  if (acc.contacts == 1) {
    // Exactly the single pair's resolve() result.
    px = acc.npx;
    py = acc.npy;
    vx = acc.nvx;
    vy = acc.nvy;
  } else if (acc.contacts > 1) {
    px = sa.px + acc.dpx;
    py = sa.py + acc.dpy;
    vx = sa.vx + acc.dvx;
    vy = sa.vy + acc.dvy;
  }
  outPosX[a] = px;
  outPosY[a] = py;
  outVelX[a] = vx;
  outVelY[a] = vy;
  contacts[a] = acc.contacts;
  aabbAsLater[a] = acc.aabbAsLater;
}

} // namespace collision_detail
