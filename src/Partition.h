// Partition.h
#pragma once
#include "Body.h"
#include <vector>
#include <cstddef>

template <typename F>
std::size_t partition(std::vector<Body>& bodies, std::size_t begin, std::size_t end, F predicate) {
    if (begin == end) return begin;
    std::size_t l = begin, r = end - 1;
    while (true) {
        while (l <= r && predicate(bodies[l])) ++l;
        while (l < r && !predicate(bodies[r])) --r;
        if (l >= r) return l;
        std::swap(bodies[l], bodies[r]);
        ++l; --r;
    }
}
