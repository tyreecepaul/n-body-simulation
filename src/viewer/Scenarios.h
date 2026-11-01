#pragma once
// Initial conditions for the viewer, and helpers that add bodies to an
// existing state. Every function returns bodies with ids 0..n-1.
#include "Body.h"
#include <cstddef>
#include <vector>

enum class Scenario { Disc, TwoGalaxies, ColdSquare };
constexpr const char *kScenarioNames[] = {"Rotating disc", "Two galaxies",
                                          "Cold square"};

std::vector<Body> makeScenario(Scenario s, std::size_t n, unsigned seed);

// Gives bodies the ids 0..n-1 in array order.
void renumber(std::vector<Body> &bodies);

// Appends one heavy body at rest at (x, y).
void addHeavyBody(std::vector<Body> &bodies, float x, float y, float mass);

// Appends `count` unit-mass bodies in a small disc around (x, y), all moving
// with velocity (vx, vy).
void addCluster(std::vector<Body> &bodies, float x, float y, float vx,
                float vy, std::size_t count, unsigned seed);
