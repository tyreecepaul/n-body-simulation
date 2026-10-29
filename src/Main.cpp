#include "Simulation.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

// Output file layout (all little-endian, native float32/uint32).
// Format version "NBS2" — breaking change from the original 6-float/body
// layout: mass and radius are invariant over the run, so they're now
// written once instead of once per frame, and a live completed_frames
// counter is kept in the header so a crashed/killed run still produces a
// file that can be read (truncated) instead of one that fails the loader's
// size check entirely.
//
//   Header (28 bytes):
//     char[4]  magic = "NBS2"
//     uint32   n_bodies
//     uint32   total_steps
//     uint32   output_every
//     uint32   num_frames        // expected frame count
//     uint32   completed_frames  // frames actually written; updated live
//     float32  dt
//
//   Then, written once:
//     float32[n_bodies]  mass
//     float32[n_bodies]  radius
//
//   Then num_frames frames, each frame = n_bodies records of 4 float32:
//     [x, y, vx, vy]
//
//   Frame 0 is the initial state (before any steps). Frame k (k >= 1)
//   is the state after step (k * output_every).
//
//   Every record, in every frame and in the mass/radius arrays, is at
//   the same array position: position i always refers to Body::id == i.
//   This is enforced by sorting sim.bodies by id immediately before each
//   write (see sortById() below), because QuadTree::build() reorders
//   sim.bodies in place every step for cache-friendly tree construction
//   — array position alone no longer identifies "the same body" across
//   frames.
//
// Total size = 28 + n_bodies * 8 + num_frames * n_bodies * 4 * 4 bytes.

namespace {
void sortById(std::vector<Body> &bodies) {
  std::sort(bodies.begin(), bodies.end(),
            [](const Body &a, const Body &b) { return a.id < b.id; });
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <n_bodies> <total_steps> <output_every> <output_path>\n";
    return 1;
  }

  std::size_t nBodies = static_cast<std::size_t>(std::stoul(argv[1]));
  std::uint32_t totalSteps = static_cast<std::uint32_t>(std::stoul(argv[2]));
  std::uint32_t outputEvery = static_cast<std::uint32_t>(std::stoul(argv[3]));
  std::string outputPath = argv[4];

  if (outputEvery == 0) {
    std::cerr << "output_every must be >= 1\n";
    return 1;
  }

  std::ofstream out(outputPath, std::ios::binary);
  if (!out) {
    std::cerr << "Failed to open output file: " << outputPath << "\n";
    return 1;
  }

  std::uint32_t n32 = static_cast<std::uint32_t>(nBodies);
  std::uint32_t numFrames =
      1 + totalSteps / outputEvery; // frame 0 + periodic frames
  std::uint32_t completedFrames = 0;

  Simulation sim(nBodies);

  char magic[4] = {'N', 'B', 'S', '2'};
  out.write(magic, sizeof(magic));
  out.write(reinterpret_cast<const char *>(&n32), sizeof(n32));
  out.write(reinterpret_cast<const char *>(&totalSteps), sizeof(totalSteps));
  out.write(reinterpret_cast<const char *>(&outputEvery), sizeof(outputEvery));
  out.write(reinterpret_cast<const char *>(&numFrames), sizeof(numFrames));

  // Remember where completed_frames lives so we can rewrite it in place
  // after every frame without disturbing the append position.
  const std::streampos completedFramesOffset = out.tellp();
  out.write(reinterpret_cast<const char *>(&completedFrames),
            sizeof(completedFrames));
  out.write(reinterpret_cast<const char *>(&sim.dt), sizeof(sim.dt));

  // uniformDisc() sorts bodies by radius from center before returning, so
  // even frame 0's array order isn't id order — sort before this first
  // write too, or the mass/radius arrays wouldn't line up with id 0..n-1.
  sortById(sim.bodies);

  // mass and radius never change after construction (collide()/iterate()
  // only touch pos/vel/acc), so write them once instead of every frame.
  for (const Body &body : sim.bodies) {
    out.write(reinterpret_cast<const char *>(&body.mass), sizeof(float));
  }
  for (const Body &body : sim.bodies) {
    out.write(reinterpret_cast<const char *>(&body.radius), sizeof(float));
  }

  auto writeFrame = [&]() {
    for (const Body &body : sim.bodies) {
      float record[4] = {body.pos.x, body.pos.y, body.vel.x, body.vel.y};
      out.write(reinterpret_cast<const char *>(record), sizeof(record));
    }
  };

  auto markFrameComplete = [&]() {
    ++completedFrames;
    const std::streampos savedPos = out.tellp();
    out.seekp(completedFramesOffset);
    out.write(reinterpret_cast<const char *>(&completedFrames),
              sizeof(completedFrames));
    out.seekp(savedPos);
  };

  writeFrame();
  markFrameComplete();

  // Timed: the whole step loop (total_time), and within it only the
  // sim.step() calls (compute_time), so frame output (the id sort and the
  // file writes) can be separated out as output_time.
  using Clock = std::chrono::steady_clock;
  double computeMs = 0.0;
  const auto loopStart = Clock::now();

  for (std::uint32_t step = 1; step <= totalSteps; ++step) {
    const auto stepStart = Clock::now();
    sim.step(); // attract() -> quadtree.build() reorders sim.bodies
    computeMs += std::chrono::duration<double, std::milli>(Clock::now() -
                                                           stepStart)
                     .count();
    if (step % outputEvery == 0) {
      sortById(sim.bodies); // restore id order before this frame's write
      writeFrame();
      markFrameComplete();
    }
  }

  const double loopMs =
      std::chrono::duration<double, std::milli>(Clock::now() - loopStart)
          .count();

  out.close();
  if (!out) {
    std::cerr << "Error while writing output file: " << outputPath << "\n";
    return 1;
  }

  std::cerr << "Done. n=" << nBodies << " steps=" << totalSteps
            << " output_every=" << outputEvery
            << " frames_written=" << completedFrames << " (expected "
            << numFrames << ") -> " << outputPath << "\n";
  std::cerr << "BH: n=" << nBodies << " threads=1"
            << " steps=" << totalSteps << " total_time=" << loopMs << " ms"
            << " avg_step=" << (totalSteps ? loopMs / totalSteps : 0.0)
            << " ms compute_time=" << computeMs << " ms"
            << " avg_compute_step="
            << (totalSteps ? computeMs / totalSteps : 0.0) << " ms"
            << " output_time=" << (loopMs - computeMs) << " ms\n";

  return 0;
}
