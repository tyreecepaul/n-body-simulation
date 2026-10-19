// naive_sim_cuda: GPU counterpart of naive_sim (src/naive/MainNaive.cpp).
// Times totalSteps steps of the naive O(N^2) simulation and prints the
// same summary line format, prefixed "CUDA".

#include "CudaNaive.cuh"

#include <cstdint>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <n_bodies> <total_steps>\n";
    return 1;
  }

  std::size_t nBodies = static_cast<std::size_t>(std::stoul(argv[1]));
  std::uint32_t totalSteps = static_cast<std::uint32_t>(std::stoul(argv[2]));

  // Setup (CUDA context, upload, allocations) happens here, outside the
  // timed region, as the CPU version's construction is.
  CudaNaiveSimulation sim(nBodies);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  for (std::uint32_t step = 0; step < totalSteps; ++step) {
    sim.step();
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsedMs = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::cerr << "CUDA naive O(N^2): n=" << nBodies << " steps=" << totalSteps
            << " total_time=" << elapsedMs << " ms"
            << " avg_step=" << (elapsedMs / totalSteps) << " ms\n";

  return 0;
}
