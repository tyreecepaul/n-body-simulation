#pragma once
// Per-node / per-body quadtree logic used by CudaQuadTree.cu's kernels.
// Plain __host__ __device__ functions, so the kernels stay thin wrappers
// and the algorithm can also be run on the host without a GPU.
//
// The tree mirrors QuadTree.cpp's layout: breadth-first nodes, the 4
// children of a node stored consecutively at `children` (0 = leaf), a
// `next` skip pointer for stackless traversal (0 = end), and each node
// owning a contiguous [start, end) range of bodies. Bodies are put in that
// order by sorting on a Morton (Z-order) code of their position within the
// root square, instead of QuadTree.cpp's in-place partitioning.

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#define NBODY_HD __host__ __device__
#else
#define NBODY_HD
#endif

namespace tree_detail {

// Morton codes interleave 32 bits per axis, so a node at depth d uses
// 2-bit digit d; nodes at kMaxDepth stay leaves however many bodies they
// hold (QuadTree.cpp has no depth limit).
constexpr int kMaxDepth = 32;

// Same parameters as Simulation's QuadTree(theta, epsilon, leafCapacity).
constexpr float kTheta = 1.0f;
constexpr float kEpsilon = 1.0f;
constexpr int kLeafCapacity = 16;

// Spreads the 32 bits of v into the even bit positions of a 64-bit word.
NBODY_HD inline uint64_t spreadBits(uint32_t v) {
  uint64_t x = v;
  x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
  x = (x | (x << 8)) & 0x00FF00FF00FF00FFull;
  x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Full;
  x = (x | (x << 2)) & 0x3333333333333333ull;
  x = (x | (x << 1)) & 0x5555555555555555ull;
  return x;
}

// Quantises a coordinate to 32 bits across [lo, lo + size].
NBODY_HD inline uint32_t quantise(float v, double lo, double scale) {
  const double q = floor((static_cast<double>(v) - lo) * scale);
  if (!(q > 0.0)) {
    return 0u; // also catches NaN
  }
  if (q >= 4294967295.0) {
    return 0xFFFFFFFFu;
  }
  return static_cast<uint32_t>(q);
}

// Each 2-bit digit is (y bit, x bit), so digit = 2*y + x: the same
// quadrant numbering as Quad::intoQuadrant (0 bottom-left, 1 bottom-right,
// 2 top-left, 3 top-right) and the same child order as QuadTree::subdivide.
NBODY_HD inline uint64_t mortonCode(float x, float y, double x0, double y0,
                                    double scale) {
  return (spreadBits(quantise(y, y0, scale)) << 1) |
         spreadBits(quantise(x, x0, scale));
}

NBODY_HD inline int digitAt(uint64_t code, int depth) {
  return static_cast<int>((code >> (62 - 2 * depth)) & 3u);
}

// First index in [lo, hi) whose digit at `depth` is >= q (codes in a
// node's range share all higher digits, so this digit is sorted).
NBODY_HD inline int firstWithDigit(const uint64_t *codes, int lo, int hi,
                                   int depth, int q) {
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (digitAt(codes[mid], depth) < q) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Structure-of-arrays node storage.
struct NodeArrays {
  float *comX, *comY;       // weighted position sum, then centre of mass
  float *mass;
  float *centerX, *centerY; // quad centre
  float *size;              // quad side length
  int *children, *next;
  int *start, *end;         // body range
};

// Body arrays read during build / traversal (already in Morton order).
struct TreeBodies {
  const float *posX, *posY, *mass;
};

NBODY_HD inline void initNode(const NodeArrays &t, int node, float cx,
                              float cy, float size, int next, int start,
                              int end) {
  t.comX[node] = 0.0f;
  t.comY[node] = 0.0f;
  t.mass[node] = 0.0f;
  t.centerX[node] = cx;
  t.centerY[node] = cy;
  t.size[node] = size;
  t.children[node] = 0;
  t.next[node] = next;
  t.start[node] = start;
  t.end[node] = end;
}

NBODY_HD inline bool shouldSplit(const NodeArrays &t, int node, int depth) {
  return t.end[node] - t.start[node] > kLeafCapacity && depth < kMaxDepth;
}

// Creates the 4 children of `node` at slots children..children+3, as
// QuadTree::subdivide does: child quads from Quad::intoQuadrant, body
// ranges split at the Morton digit boundaries, `next` pointers chained
// with the last child inheriting the parent's next.
NBODY_HD inline void splitNode(const NodeArrays &t, int node, int children,
                               int depth, const uint64_t *codes) {
  const int start = t.start[node];
  const int end = t.end[node];
  int bound[5];
  bound[0] = start;
  bound[4] = end;
  for (int q = 1; q < 4; ++q) {
    bound[q] = firstWithDigit(codes, bound[q - 1], end, depth, q);
  }

  const float childSize = t.size[node] * 0.5f;
  const float cx = t.centerX[node];
  const float cy = t.centerY[node];
  const int parentNext = t.next[node];
  t.children[node] = children;
  for (int q = 0; q < 4; ++q) {
    const float ccx = cx + ((q & 1) - 0.5f) * childSize;
    const float ccy = cy + ((q >> 1) - 0.5f) * childSize;
    const int next = q < 3 ? children + q + 1 : parentNext;
    initNode(t, children + q, ccx, ccy, childSize, next, bound[q],
             bound[q + 1]);
  }
}

// Bottom-up pass for one node (children must already be done): leaves sum
// their bodies, branches their 4 children. Sums are mass-weighted
// positions; finishNode() divides by mass afterwards (QuadTree::propagate).
NBODY_HD inline void sumNode(const NodeArrays &t, int node,
                             const TreeBodies &b) {
  float px = 0.0f, py = 0.0f, m = 0.0f;
  const int c = t.children[node];
  if (c == 0) {
    for (int i = t.start[node]; i < t.end[node]; ++i) {
      px += b.posX[i] * b.mass[i];
      py += b.posY[i] * b.mass[i];
      m += b.mass[i];
    }
  } else {
    px = t.comX[c] + t.comX[c + 1] + t.comX[c + 2] + t.comX[c + 3];
    py = t.comY[c] + t.comY[c + 1] + t.comY[c + 2] + t.comY[c + 3];
    m = t.mass[c] + t.mass[c + 1] + t.mass[c + 2] + t.mass[c + 3];
  }
  t.comX[node] = px;
  t.comY[node] = py;
  t.mass[node] = m;
}

NBODY_HD inline void finishNode(const NodeArrays &t, int node) {
  const float m = fmaxf(t.mass[node], FLT_MIN);
  t.comX[node] /= m;
  t.comY[node] /= m;
}

// QuadTree.cpp's fastInvSqrt (bit-level initial guess + one Newton step).
NBODY_HD inline float fastInvSqrt(float x) {
  const float xhalf = 0.5f * x;
  uint32_t i;
#ifdef __CUDA_ARCH__
  i = __float_as_uint(x);
#else
  memcpy(&i, &x, sizeof i);
#endif
  i = 0x5f3759dfu - (i >> 1);
#ifdef __CUDA_ARCH__
  x = __uint_as_float(i);
#else
  memcpy(&x, &i, sizeof x);
#endif
  return x * (1.5f - xhalf * x * x);
}

// Port of QuadTree::acceleration: stackless walk using `next`, opening
// criterion size^2 < d^2 * theta^2, same softened force law.
NBODY_HD inline void accelerationAt(float x, float y, const NodeArrays &t,
                                    const TreeBodies &b, float &ax,
                                    float &ay) {
  const float tSq = kTheta * kTheta;
  const float eSq = kEpsilon * kEpsilon;
  float accX = 0.0f;
  float accY = 0.0f;
  int node = 0;
  while (true) {
    const float dx = t.comX[node] - x;
    const float dy = t.comY[node] - y;
    const float dSq = dx * dx + dy * dy;
    const float size = t.size[node];

    if (size * size < dSq * tSq) {
      // Far enough away: centre-of-mass approximation.
      const float m = t.mass[node];
      if (m > 0.0f) {
        const float factor =
            fminf(m * fastInvSqrt(dSq) * (1.0f / (dSq + eSq)), FLT_MAX);
        accX += dx * factor;
        accY += dy * factor;
      }
    } else if (t.children[node] == 0) {
      // Close leaf: sum its bodies individually.
      for (int i = t.start[node]; i < t.end[node]; ++i) {
        const float bx = b.posX[i] - x;
        const float by = b.posY[i] - y;
        const float bSq = bx * bx + by * by;
        const float factor = fminf(
            b.mass[i] * fastInvSqrt(bSq) * (1.0f / (bSq + eSq)), FLT_MAX);
        accX += bx * factor;
        accY += by * factor;
      }
    } else {
      node = t.children[node]; // close branch: descend
      continue;
    }
    node = t.next[node];
    if (node == 0) {
      break;
    }
  }
  ax = accX;
  ay = accY;
}

} // namespace tree_detail
