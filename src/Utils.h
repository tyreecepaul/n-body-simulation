#pragma once
#include "Body.h"
#include <cstddef>
#include <vector>

// Generates n bodies arranged in a rotating disc, with one heavy
// central body. Port of Rust's utils::uniform_disc.
std::vector<Body> uniformDisc(std::size_t n);
