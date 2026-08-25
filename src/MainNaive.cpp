#include "NaiveSimulation.h"
#include <chrono>
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

  NaiveSimulation sim(nBodies);

  auto start = std::chrono::high_resolution_clock::now();

  for (std::uint32_t step = 0; step < totalSteps; ++step) {
    sim.step();
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsedMs =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cerr << "Naive O(N^2): n=" << nBodies << " steps=" << totalSteps
            << " total_time=" << elapsedMs << " ms"
            << " avg_step=" << (elapsedMs / totalSteps) << " ms\n";

  return 0;
}
