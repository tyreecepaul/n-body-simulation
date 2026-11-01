#pragma once
#include "DeviceBodies.cuh"
#include "TreeKernels.cuh"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// GPU Barnes-Hut quadtree, rebuilt every step (see TreeKernels.cuh for the
// node layout, which mirrors QuadTree.cpp's).
//
// build(): bounding square (CUB min/max reductions) -> Morton code per body
// -> stable CUB radix sort -> every body array permuted into Morton order
// (so each node's bodies are contiguous, like QuadTree.cpp's in-place
// partition) -> breadth-first, one kernel per level: nodes with more than
// 16 bodies split, children placed at levelEnd + 4 * (exclusive-scan rank)
// -> bottom-up mass/centre-of-mass sums, one kernel per level in reverse.
//
// computeForces(): one thread per body walks the tree exactly as
// QuadTree::acceleration does (theta = 1, epsilon = 1).
class CudaQuadTree {
public:
  explicit CudaQuadTree(std::size_t n);

  // Reorders all of b's arrays into Morton order and builds the tree.
  void build(DeviceBodies &b);

  // Barnes-Hut accelerations into b.accX/b.accY; b must be as build()
  // left it.
  void computeForces(DeviceBodies &b);

  // Fetched from the device on first call after a cooperative build.
  int nodeCount() const;
  int levelCount() const;

  // The default build is one cooperative kernel (no host round-trips).
  // false selects the host-driven per-level loop, which produces the same
  // tree; used automatically on GPUs without cooperative launch.
  void setUseCooperativeBuild(bool use);
  bool usesCooperativeBuild() const { return cooperative_; }

  // Host copy of the node arrays, for checks.
  struct HostNodes {
    std::vector<float> comX, comY, mass, centerX, centerY, size;
    std::vector<int> children, next, start, end;
  };
  void download(HostNodes &out) const;

private:
  void buildCooperative(const tree_detail::NodeArrays &t,
                        const tree_detail::TreeBodies &tb);
  void buildHostLoop(const tree_detail::NodeArrays &t,
                     const tree_detail::TreeBodies &tb);
  void syncCounts() const;

  std::size_t n_ = 0;
  std::size_t capacity_ = 0; // node capacity (upper bound, see .cu)
  mutable int nodeCount_ = 0;
  mutable int levelCount_ = 0;
  mutable bool countsOnDevice_ = false;

  bool cooperative_ = false;
  int coopGrid_ = 0;
  DeviceBuffer<int> blockTotals_; // per-block split counts, then offsets
  DeviceBuffer<int> levelTable_;  // [begin, end) per depth
  DeviceBuffer<int> info_;        // see kInfo* in the .cu

  DeviceBuffer<float> bounds_; // minX, maxX, minY, maxY
  DeviceBuffer<std::uint64_t> codes_, codesSorted_;
  DeviceBuffer<int> perm_, permSorted_;
  DeviceBuffer<float> gatherF_;
  DeviceBuffer<std::uint32_t> gatherU_;
  DeviceBuffer<int> flags_, ranks_;
  DeviceBuffer<unsigned char> temp_; // shared CUB scratch
  std::size_t tempBytes_ = 0;

  DeviceBuffer<float> comX_, comY_, mass_, centerX_, centerY_, size_;
  DeviceBuffer<int> children_, next_, start_, end_;
};
