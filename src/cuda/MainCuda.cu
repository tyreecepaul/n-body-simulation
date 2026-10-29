// sim_cuda: GPU counterpart of sim (src/Main.cpp). Same command line and
// the same NBS2 output file (see Main.cpp / load_sim.py for the layout),
// so load_sim.py and run.py read it unchanged.
//
// Bodies stay on the GPU; on output frames positions/velocities/ids are
// copied back and placed at their id slot (array position i in the file is
// always Body::id i), then written with one call. Unlike Main.cpp, this
// doesn't reorder the simulation's own arrays.

#include "CudaSimulation.cuh"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
  std::uint32_t numFrames = 1 + totalSteps / outputEvery;
  std::uint32_t completedFrames = 0;

  CudaSimulation sim(nBodies);

  char magic[4] = {'N', 'B', 'S', '2'};
  out.write(magic, sizeof(magic));
  out.write(reinterpret_cast<const char *>(&n32), sizeof(n32));
  out.write(reinterpret_cast<const char *>(&totalSteps), sizeof(totalSteps));
  out.write(reinterpret_cast<const char *>(&outputEvery), sizeof(outputEvery));
  out.write(reinterpret_cast<const char *>(&numFrames), sizeof(numFrames));
  const std::streampos completedFramesOffset = out.tellp();
  out.write(reinterpret_cast<const char *>(&completedFrames),
            sizeof(completedFrames));
  out.write(reinterpret_cast<const char *>(&sim.dt), sizeof(sim.dt));

  // mass and radius never change: write them once, in id order.
  std::vector<std::uint32_t> ids;
  std::vector<float> mass, radius, byId(nBodies);
  sim.bodies.id.download(ids);
  sim.bodies.mass.download(mass);
  sim.bodies.radius.download(radius);
  for (std::size_t i = 0; i < nBodies; ++i) {
    byId[ids[i]] = mass[i];
  }
  out.write(reinterpret_cast<const char *>(byId.data()),
            static_cast<std::streamsize>(nBodies * sizeof(float)));
  for (std::size_t i = 0; i < nBodies; ++i) {
    byId[ids[i]] = radius[i];
  }
  out.write(reinterpret_cast<const char *>(byId.data()),
            static_cast<std::streamsize>(nBodies * sizeof(float)));

  std::vector<float> px, py, vx, vy;
  std::vector<float> frame(nBodies * 4);
  auto writeFrame = [&]() {
    sim.bodies.posX.download(px);
    sim.bodies.posY.download(py);
    sim.bodies.velX.download(vx);
    sim.bodies.velY.download(vy);
    sim.bodies.id.download(ids);
    for (std::size_t i = 0; i < nBodies; ++i) {
      float *record = &frame[static_cast<std::size_t>(ids[i]) * 4];
      record[0] = px[i];
      record[1] = py[i];
      record[2] = vx[i];
      record[3] = vy[i];
    }
    out.write(reinterpret_cast<const char *>(frame.data()),
              static_cast<std::streamsize>(frame.size() * sizeof(float)));
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

  // Timed: the step loop including output frames (like `time ./sim`, but
  // without CUDA start-up and the initial upload).
  //
  // Compute time alone is measured with CUDA events around each run of
  // steps between output frames: everything is on the default stream and
  // the frame downloads come after the segment's stop event, so it covers
  // the steps' GPU work (and any host round-trips inside step()) but not
  // the copy-back or the file writes.
  cudaEvent_t segmentStart, segmentStop;
  CUDA_CHECK(cudaEventCreate(&segmentStart));
  CUDA_CHECK(cudaEventCreate(&segmentStop));
  double computeMs = 0.0;
  auto endSegment = [&]() {
    CUDA_CHECK(cudaEventRecord(segmentStop));
    CUDA_CHECK(cudaEventSynchronize(segmentStop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, segmentStart, segmentStop));
    computeMs += ms;
  };

  CUDA_CHECK(cudaDeviceSynchronize());
  const auto start = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(segmentStart));

  for (std::uint32_t step = 1; step <= totalSteps; ++step) {
    sim.step();
    if (step % outputEvery == 0) {
      endSegment();
      writeFrame();
      markFrameComplete();
      CUDA_CHECK(cudaEventRecord(segmentStart));
    }
  }
  if (totalSteps % outputEvery != 0) {
    endSegment(); // steps after the last frame
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  const double elapsedMs = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start)
                               .count();
  CUDA_CHECK(cudaEventDestroy(segmentStart));
  CUDA_CHECK(cudaEventDestroy(segmentStop));

  out.close();
  if (!out) {
    std::cerr << "Error while writing output file: " << outputPath << "\n";
    return 1;
  }

  std::cerr << "Done. n=" << nBodies << " steps=" << totalSteps
            << " output_every=" << outputEvery
            << " frames_written=" << completedFrames << " (expected "
            << numFrames << ") -> " << outputPath << "\n";
  std::cerr << "CUDA BH: n=" << nBodies << " steps=" << totalSteps
            << " total_time=" << elapsedMs << " ms"
            << " avg_step=" << (totalSteps ? elapsedMs / totalSteps : 0.0)
            << " ms compute_time=" << computeMs << " ms"
            << " avg_compute_step="
            << (totalSteps ? computeMs / totalSteps : 0.0) << " ms"
            << " output_time=" << (elapsedMs - computeMs) << " ms\n";

  return 0;
}
