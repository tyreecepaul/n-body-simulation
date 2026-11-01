#pragma once
#include "Body.h"
#include <cstddef>
#include <vector>

// Generates n bodies arranged in a rotating disc, with one heavy
// central body. Port of Rust's utils::uniform_disc. The simulations all
// use seed 0; other seeds give different discs (used by the viewer).
std::vector<Body> uniformDisc(std::size_t n, unsigned seed = 0);
