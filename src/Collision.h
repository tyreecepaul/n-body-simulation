#pragma once
#include "Body.h"
#include <vector>

// Detects and resolves overlapping bodies. Shared by Simulation and
// NaiveSimulation so collision handling is identical by construction and
// any performance/behaviour difference between them comes only from
// attract().
void collide(std::vector<Body> &bodies);
