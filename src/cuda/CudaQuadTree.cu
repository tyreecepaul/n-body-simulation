#include "CudaQuadTree.cuh"
#include "TreeKernels.cuh"

#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace {
using namespace tree_detail;

constexpr int kBlock = 256;

int blocksFor(std::size_t n) {
  return static_cast<int>((n + kBlock - 1) / kBlock);
}

// Root square, exactly as Quad::newContaining computes it.
struct RootSquare {
  float cx, cy, size;
};

__device__ inline RootSquare rootSquare(const float *bounds) {
  const float minX = bounds[0], maxX = bounds[1];
  const float minY = bounds[2], maxY = bounds[3];
  return RootSquare{(minX + maxX) * 0.5f, (minY + maxY) * 0.5f,
                    fmaxf(maxX - minX, maxY - minY)};
}

__global__ void mortonKernel(int n, const float *__restrict__ bounds,
                             const float *__restrict__ posX,
                             const float *__restrict__ posY,
                             uint64_t *__restrict__ codes,
                             int *__restrict__ perm) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const RootSquare root = rootSquare(bounds);
  const double x0 = static_cast<double>(root.cx) - 0.5 * root.size;
  const double y0 = static_cast<double>(root.cy) - 0.5 * root.size;
  const double scale = root.size > 0.0f ? 4294967296.0 / root.size : 0.0;
  codes[i] = mortonCode(posX[i], posY[i], x0, y0, scale);
  perm[i] = i;
}

template <typename T>
__global__ void gatherKernel(int n, const int *__restrict__ perm,
                             const T *__restrict__ in, T *__restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[perm[i]];
  }
}

__global__ void rootKernel(NodeArrays t, const float *__restrict__ bounds,
                           int n) {
  const RootSquare root = rootSquare(bounds);
  initNode(t, 0, root.cx, root.cy, root.size, /*next=*/0, 0, n);
}

// flags[k] = does node levelBegin + k split; flags[levelSize] = 0 so the
// exclusive scan's last entry is the number of splitting nodes.
__global__ void flagKernel(NodeArrays t, int levelBegin, int levelSize,
                           int depth, int *__restrict__ flags) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < levelSize) {
    flags[k] = shouldSplit(t, levelBegin + k, depth) ? 1 : 0;
  } else if (k == levelSize) {
    flags[k] = 0;
  }
}

__global__ void splitKernel(NodeArrays t, int levelBegin, int levelSize,
                            int levelEnd, int depth,
                            const int *__restrict__ flags,
                            const int *__restrict__ ranks,
                            const uint64_t *__restrict__ codes) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < levelSize && flags[k]) {
    splitNode(t, levelBegin + k, levelEnd + 4 * ranks[k], depth, codes);
  }
}

__global__ void sumKernel(NodeArrays t, int levelBegin, int levelSize,
                          TreeBodies b) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < levelSize) {
    sumNode(t, levelBegin + k, b);
  }
}

__global__ void finishKernel(NodeArrays t, int nodeCount) {
  const int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node < nodeCount) {
    finishNode(t, node);
  }
}

// One thread per body, in Morton order, so neighbouring threads in a warp
// walk nearly the same path through the tree.
__global__ void forceKernel(int n, NodeArrays t, TreeBodies b,
                            float *__restrict__ accX,
                            float *__restrict__ accY) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float ax, ay;
  accelerationAt(b.posX[i], b.posY[i], t, b, ax, ay);
  accX[i] = ax;
  accY[i] = ay;
}

// Layout of the small device-side status block written by buildTreeKernel.
enum : int {
  kInfoLevels = 0,    // number of levels (depths) in the tree
  kInfoNodes = 1,     // total node count
  kInfoOverflow = 2,  // non-zero if node capacity would have been exceeded
  kInfoSplitting = 3, // splitting nodes in the current level (internal)
  kInfoSize = 4
};

constexpr int kBuildBlock = 256;

// The whole top-down build and bottom-up sums in one cooperative launch,
// with grid-wide barriers between phases instead of a host round-trip per
// level. Per level:
//   1. each block takes a contiguous chunk of the level's nodes and ranks
//      its splitting nodes (block-wide exclusive scan);
//   2. block 0 scans the per-block totals into per-block offsets;
//   3. each splitting node's children go at levelEnd + 4 * global rank.
// Contiguous chunks in block order make the global rank the same as an
// exclusive scan over the whole level, so the node layout (and therefore
// the tree) is identical to the host-driven loop's.
__global__ void __launch_bounds__(kBuildBlock)
    buildTreeKernel(NodeArrays t, TreeBodies b,
                    const uint64_t *__restrict__ codes, int capacity,
                    int *__restrict__ localRanks, int *__restrict__ blockTotals,
                    int *__restrict__ levelTable, int *info) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  using BlockScan = cub::BlockScan<int, kBuildBlock>;
  __shared__ typename BlockScan::TempStorage scanTemp;
  __shared__ int carry;

  const int blocks = static_cast<int>(gridDim.x);
  const int block = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x);

  int levelBegin = 0;
  int levelEnd = 1;
  int depth = 0;
  while (true) {
    const int levelSize = levelEnd - levelBegin;
    const int chunk = (levelSize + blocks - 1) / blocks;
    const int lo = min(levelSize, block * chunk);
    const int hi = min(levelSize, lo + chunk);

    // 1. Local ranks of splitting nodes within this block's chunk.
    if (lane == 0) {
      carry = 0;
    }
    __syncthreads();
    for (int base = lo; base < hi; base += kBuildBlock) {
      const int k = base + lane;
      const int flag = k < hi && shouldSplit(t, levelBegin + k, depth) ? 1 : 0;
      int rank, tileTotal;
      BlockScan(scanTemp).ExclusiveSum(flag, rank, tileTotal);
      if (k < hi) {
        localRanks[k] = flag ? carry + rank : -1;
      }
      __syncthreads();
      if (lane == 0) {
        carry += tileTotal;
      }
      __syncthreads();
    }
    if (lane == 0) {
      blockTotals[block] = carry;
    }
    grid.sync();

    // 2. Block 0: per-block offsets, level bookkeeping, capacity check.
    if (block == 0) {
      if (lane == 0) {
        carry = 0;
      }
      __syncthreads();
      for (int base = 0; base < blocks; base += kBuildBlock) {
        const int i = base + lane;
        const int v = i < blocks ? blockTotals[i] : 0;
        int exclusive, tileTotal;
        BlockScan(scanTemp).ExclusiveSum(v, exclusive, tileTotal);
        if (i < blocks) {
          blockTotals[i] = carry + exclusive;
        }
        __syncthreads();
        if (lane == 0) {
          carry += tileTotal;
        }
        __syncthreads();
      }
      if (lane == 0) {
        int splitting = carry;
        levelTable[2 * depth] = levelBegin;
        levelTable[2 * depth + 1] = levelEnd;
        info[kInfoLevels] = depth + 1;
        if (static_cast<long long>(levelEnd) + 4ll * splitting > capacity) {
          info[kInfoOverflow] = 1;
          splitting = 0; // stop building; the host reports the overflow
        }
        info[kInfoSplitting] = splitting;
      }
    }
    grid.sync();

    const int splitting = *static_cast<volatile int *>(&info[kInfoSplitting]);
    if (splitting == 0) {
      break; // same value in every block, so all leave together
    }

    // 3. Create the children of this chunk's splitting nodes.
    const int offset = blockTotals[block];
    for (int k = lo + lane; k < hi; k += kBuildBlock) {
      const int r = localRanks[k];
      if (r >= 0) {
        splitNode(t, levelBegin + k, levelEnd + 4 * (offset + r), depth,
                  codes);
      }
    }
    levelBegin = levelEnd;
    levelEnd += 4 * splitting;
    ++depth;
    grid.sync();
  }
  if (block == 0 && lane == 0) {
    info[kInfoNodes] = levelEnd;
  }

  // Bottom-up: deepest level first, so children are summed before parents.
  const int stride = blocks * kBuildBlock;
  const int tid = block * kBuildBlock + lane;
  for (int level = depth; level >= 0; --level) {
    const int begin = levelTable[2 * level];
    const int end = levelTable[2 * level + 1];
    for (int node = begin + tid; node < end; node += stride) {
      sumNode(t, node, b);
    }
    grid.sync();
  }
  for (int node = tid; node < levelEnd; node += stride) {
    finishNode(t, node);
  }
}

template <typename T>
void permute(DeviceBuffer<T> &array, DeviceBuffer<T> &scratch,
             const DeviceBuffer<int> &perm, int n) {
  gatherKernel<T><<<blocksFor(n), kBlock>>>(n, perm.data(), array.data(),
                                            scratch.data());
  CUDA_CHECK_LAUNCH();
  array.swap(scratch);
}
} // namespace

CudaQuadTree::CudaQuadTree(std::size_t n) : n_(n) {
  // Node bound: a splitting node holds > 16 bodies and nodes at one depth
  // hold disjoint bodies, so each depth has <= n/17 splitting nodes; with
  // at most kMaxDepth depths, nodes <= 1 + 4 * kMaxDepth * n / 17 < 8n + 8.
  capacity_ = 8 * n_ + 8;

  bounds_.resize(4);
  codes_.resize(n_);
  codesSorted_.resize(n_);
  perm_.resize(n_);
  permSorted_.resize(n_);
  gatherF_.resize(n_);
  gatherU_.resize(n_);
  flags_.resize(capacity_ + 1);
  ranks_.resize(capacity_ + 1);

  comX_.resize(capacity_);
  comY_.resize(capacity_);
  mass_.resize(capacity_);
  centerX_.resize(capacity_);
  centerY_.resize(capacity_);
  size_.resize(capacity_);
  children_.resize(capacity_);
  next_.resize(capacity_);
  start_.resize(capacity_);
  end_.resize(capacity_);

  // One scratch buffer, sized for the largest CUB call (null = size query).
  const int ni = static_cast<int>(n_);
  std::size_t bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::Min(nullptr, bytes, bounds_.data(),
                                    bounds_.data(), ni));
  tempBytes_ = std::max(tempBytes_, bytes);
  CUDA_CHECK(cub::DeviceReduce::Max(nullptr, bytes, bounds_.data(),
                                    bounds_.data(), ni));
  tempBytes_ = std::max(tempBytes_, bytes);
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, bytes, codes_.data(), codesSorted_.data(), perm_.data(),
      permSorted_.data(), ni));
  tempBytes_ = std::max(tempBytes_, bytes);
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, flags_.data(),
                                           ranks_.data(),
                                           static_cast<int>(capacity_ + 1)));
  tempBytes_ = std::max(tempBytes_, bytes);
  temp_.resize(std::max<std::size_t>(tempBytes_, 1));

  // Cooperative build: all blocks must be resident at once, so size the
  // grid from occupancy (at most 2 blocks per SM: more only adds barrier
  // cost for work this small).
  int device = 0;
  int cooperative = 0;
  int sms = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                                    device));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                    device));
  int blocksPerSm = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocksPerSm, buildTreeKernel, kBuildBlock, 0));
  coopGrid_ = sms * std::min(blocksPerSm, 2);
  cooperative_ = cooperative != 0 && coopGrid_ > 0;
  blockTotals_.resize(std::max(coopGrid_, 1));
  levelTable_.resize(2 * (kMaxDepth + 1));
  info_.resize(kInfoSize);
}

void CudaQuadTree::setUseCooperativeBuild(bool use) {
  int device = 0;
  int cooperative = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch,
                                    device));
  cooperative_ = use && cooperative != 0 && coopGrid_ > 0;
}

int CudaQuadTree::nodeCount() const {
  syncCounts();
  return nodeCount_;
}

int CudaQuadTree::levelCount() const {
  syncCounts();
  return levelCount_;
}

// Counts from the cooperative build live on the device; fetch them (and
// check for capacity overflow) only when someone asks.
void CudaQuadTree::syncCounts() const {
  if (!countsOnDevice_) {
    return;
  }
  int info[kInfoSize];
  CUDA_CHECK(cudaMemcpy(info, info_.data(), sizeof info,
                        cudaMemcpyDeviceToHost));
  if (info[kInfoOverflow]) {
    std::fprintf(stderr, "CudaQuadTree: node capacity %zu exceeded\n",
                 capacity_);
    std::exit(1);
  }
  levelCount_ = info[kInfoLevels];
  nodeCount_ = info[kInfoNodes];
  countsOnDevice_ = false;
}

void CudaQuadTree::build(DeviceBodies &b) {
  const int n = static_cast<int>(n_);
  if (n == 0) {
    nodeCount_ = 0;
    levelCount_ = 0;
    countsOnDevice_ = false;
    return;
  }
  // CUB calls get the shared scratch buffer and its full size each time.
  std::size_t bytes = tempBytes_;

  // Bounding box -> bounds_ = {minX, maxX, minY, maxY}.
  CUDA_CHECK(cub::DeviceReduce::Min(temp_.data(), bytes, b.posX.data(),
                                    bounds_.data() + 0, n));
  bytes = tempBytes_;
  CUDA_CHECK(cub::DeviceReduce::Max(temp_.data(), bytes, b.posX.data(),
                                    bounds_.data() + 1, n));
  bytes = tempBytes_;
  CUDA_CHECK(cub::DeviceReduce::Min(temp_.data(), bytes, b.posY.data(),
                                    bounds_.data() + 2, n));
  bytes = tempBytes_;
  CUDA_CHECK(cub::DeviceReduce::Max(temp_.data(), bytes, b.posY.data(),
                                    bounds_.data() + 3, n));

  // Morton order. The sort is stable, so equal codes keep index order and
  // the result is deterministic.
  mortonKernel<<<blocksFor(n), kBlock>>>(n, bounds_.data(), b.posX.data(),
                                         b.posY.data(), codes_.data(),
                                         perm_.data());
  CUDA_CHECK_LAUNCH();
  bytes = tempBytes_;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp_.data(), bytes, codes_.data(), codesSorted_.data(), perm_.data(),
      permSorted_.data(), n));
  permute(b.posX, gatherF_, permSorted_, n);
  permute(b.posY, gatherF_, permSorted_, n);
  permute(b.velX, gatherF_, permSorted_, n);
  permute(b.velY, gatherF_, permSorted_, n);
  permute(b.mass, gatherF_, permSorted_, n);
  permute(b.radius, gatherF_, permSorted_, n);
  permute(b.id, gatherU_, permSorted_, n);

  const NodeArrays t{comX_.data(),    comY_.data(),     mass_.data(),
                     centerX_.data(), centerY_.data(),  size_.data(),
                     children_.data(), next_.data(),    start_.data(),
                     end_.data()};
  rootKernel<<<1, 1>>>(t, bounds_.data(), n);
  CUDA_CHECK_LAUNCH();

  const TreeBodies tb{b.posX.data(), b.posY.data(), b.mass.data()};
  if (cooperative_) {
    buildCooperative(t, tb);
  } else {
    buildHostLoop(t, tb);
  }
}

void CudaQuadTree::buildCooperative(const NodeArrays &t, const TreeBodies &tb) {
  CUDA_CHECK(cudaMemsetAsync(info_.data(), 0, kInfoSize * sizeof(int)));
  NodeArrays nodes = t;
  TreeBodies bodies = tb;
  const uint64_t *codes = codesSorted_.data();
  int capacity = static_cast<int>(capacity_);
  int *localRanks = ranks_.data();
  int *blockTotals = blockTotals_.data();
  int *levelTable = levelTable_.data();
  int *info = info_.data();
  void *args[] = {&nodes,       &bodies,     &codes,      &capacity,
                  &localRanks,  &blockTotals, &levelTable, &info};
  CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(buildTreeKernel),
                                         dim3(coopGrid_), dim3(kBuildBlock),
                                         args, 0, nullptr));
  countsOnDevice_ = true; // nodeCount()/levelCount() fetch lazily
}

// Host-driven fallback (and reference for the cooperative build): one
// level per iteration, with a 4-byte copy per level so the host can size
// the next one.
void CudaQuadTree::buildHostLoop(const NodeArrays &t, const TreeBodies &tb) {
  std::vector<std::pair<int, int>> levels;
  int levelBegin = 0;
  int levelEnd = 1;
  for (int depth = 0; levelBegin < levelEnd; ++depth) {
    levels.emplace_back(levelBegin, levelEnd);
    const int levelSize = levelEnd - levelBegin;

    flagKernel<<<blocksFor(levelSize + 1), kBlock>>>(t, levelBegin,
                                                     levelSize, depth,
                                                     flags_.data());
    CUDA_CHECK_LAUNCH();
    std::size_t bytes = tempBytes_;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp_.data(), bytes,
                                             flags_.data(), ranks_.data(),
                                             levelSize + 1));
    int splitting = 0;
    CUDA_CHECK(cudaMemcpy(&splitting, ranks_.data() + levelSize, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (splitting == 0) {
      break;
    }
    if (static_cast<std::size_t>(levelEnd) + 4u * splitting > capacity_) {
      std::fprintf(stderr, "CudaQuadTree: node capacity %zu exceeded\n",
                   capacity_);
      std::exit(1);
    }
    splitKernel<<<blocksFor(levelSize), kBlock>>>(
        t, levelBegin, levelSize, levelEnd, depth, flags_.data(),
        ranks_.data(), codesSorted_.data());
    CUDA_CHECK_LAUNCH();
    levelBegin = levelEnd;
    levelEnd += 4 * splitting;
  }
  nodeCount_ = levelEnd;
  levelCount_ = static_cast<int>(levels.size());
  countsOnDevice_ = false;

  // Bottom-up: deepest level first, so children are summed before parents.
  for (auto it = levels.rbegin(); it != levels.rend(); ++it) {
    const int size = it->second - it->first;
    sumKernel<<<blocksFor(size), kBlock>>>(t, it->first, size, tb);
    CUDA_CHECK_LAUNCH();
  }
  finishKernel<<<blocksFor(nodeCount_), kBlock>>>(t, nodeCount_);
  CUDA_CHECK_LAUNCH();
}

void CudaQuadTree::computeForces(DeviceBodies &b) {
  const int n = static_cast<int>(n_);
  if (n == 0) {
    return;
  }
  const NodeArrays t{comX_.data(),    comY_.data(),     mass_.data(),
                     centerX_.data(), centerY_.data(),  size_.data(),
                     children_.data(), next_.data(),    start_.data(),
                     end_.data()};
  const TreeBodies tb{b.posX.data(), b.posY.data(), b.mass.data()};
  forceKernel<<<blocksFor(n), kBlock>>>(n, t, tb, b.accX.data(),
                                        b.accY.data());
  CUDA_CHECK_LAUNCH();
}

void CudaQuadTree::download(HostNodes &out) const {
  const std::size_t count = static_cast<std::size_t>(nodeCount());
  auto take = [count](const auto &buffer, auto &host) {
    buffer.download(host);
    host.resize(count);
  };
  take(comX_, out.comX);
  take(comY_, out.comY);
  take(mass_, out.mass);
  take(centerX_, out.centerX);
  take(centerY_, out.centerY);
  take(size_, out.size);
  take(children_, out.children);
  take(next_, out.next);
  take(start_, out.start);
  take(end_, out.end);
}
