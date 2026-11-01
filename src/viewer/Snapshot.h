#pragma once
// Single-frame NBS2 files for the viewer: the same layout sim and sim_cuda
// write (see src/Main.cpp / load_sim.py), so load_sim.py and run.py read
// snapshots, and the viewer can continue any sim/sim_cuda output.
#include "Body.h"
#include <string>
#include <vector>

// Writes `bodies` as a one-frame NBS2 file. Their ids must be exactly
// 0..n-1 (any order); the file stores them in id order.
bool saveSnapshot(const std::string &path, const std::vector<Body> &bodies,
                  float dt, std::string &error);

// Reads the last completed frame of an NBS2 file (positions, velocities)
// with its masses and radii. Ids are 0..n-1 in file order; dt is the
// file's time step.
bool loadSnapshot(const std::string &path, std::vector<Body> &bodies,
                  float &dt, std::string &error);
