#pragma once
#include "DeviceBodies.cuh"

#include <cstdint>
#include <vector>

// GPU collision detection + resolution, shared by the CUDA Barnes-Hut and
// naive simulations (as Collision.cpp is shared on the CPU).
//
// Broadphase follows Collision.cpp: a uniform grid with cell = 1.01 x the
// median box width, boxes binned by their min corner, and boxes wider than
// a cell ("large", e.g. the heavy central body) tested against everything.
//
// Resolution differs deliberately from the CPU. Collision.cpp resolves
// pairs one after another, each seeing the previous pair's result; that is
// inherently serial. Here every pair is resolved from the start-of-collide
// state with the same resolve() math and the same orientation (the pair's
// "later" box by (minX, index) is body 1), and each body combines its own
// side of all its contacts:
//   - no contacts:   unchanged;
//   - one contact:   exactly the resolve() result for that pair;
//   - k > 1 contacts: start state + the sum of each pair's change.
// So isolated pairs match the CPU up to float rounding, and simultaneous
// multi-body contacts are applied together instead of sequentially. No
// float atomics are used, so results are deterministic run to run.
class CudaCollider {
public:
  // Radii never change during a run, so the grid cell size is fixed here
  // from the bodies' radii. Buffers are sized for b.n bodies.
  explicit CudaCollider(const DeviceBodies &b);

  // Detects and resolves collisions, updating b's positions and velocities.
  void collide(DeviceBodies &b);

  // Diagnostics for the most recent collide() (copies from the device).
  // aabbPairs: unordered pairs whose boxes overlap (the CPU broadphase's
  // candidate set). contacts[i]: pairs body i actually collided in.
  std::int64_t lastAabbPairs() const;
  void lastContacts(std::vector<int> &contacts) const;

  float cellSize() const { return cell_; }

  // Large boxes (e.g. the central body) are scanned against all bodies.
  // By default that scan is split across many blocks and the partial sums
  // combined in a fixed order; false selects the single-block reference
  // path (same results up to float summation order; used by checks).
  void setSplitLargeScan(bool split) { splitLargeScan_ = split; }

private:
  std::size_t n_ = 0;
  float cell_ = 1.0f;

  DeviceBuffer<unsigned long long> keys_, keysSorted_;
  DeviceBuffer<int> index_, indexSorted_;
  DeviceBuffer<unsigned char> sortTemp_;
  std::size_t sortTempBytes_ = 0;
  DeviceBuffer<int> largeCount_; // single counter
  DeviceBuffer<float> outPosX_, outPosY_, outVelX_, outVelY_;
  DeviceBuffer<int> contacts_, aabbAsLater_;
  bool splitLargeScan_ = true;
  DeviceBuffer<unsigned char> largePartials_; // Accum per (large, chunk)
};
