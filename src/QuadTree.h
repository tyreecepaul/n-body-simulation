#pragma once

#include "Body.h"
#include <cstddef>
#include <vector>

struct Quad {
    Vec2 center;
    float size;

    static Quad newContaining(const std::vector<Body> &bodies);

    // Small functions defined in the header so they can be inlined.
    inline Quad intoQuadrant(int quadrant) const {
        Quad result = *this;
        result.size *= 0.5f;

        result.center.x +=
            ((quadrant & 1) - 0.5f) * result.size;

        result.center.y +=
            ((quadrant >> 1) - 0.5f) * result.size;

        return result;
    }

    inline void subdivide(Quad out[4]) const {
        out[0] = intoQuadrant(0);
        out[1] = intoQuadrant(1);
        out[2] = intoQuadrant(2);
        out[3] = intoQuadrant(3);
    }
};

struct Node {
    int children = 0;
    int next = 0;
    Vec2 pos{};
    float mass = 0.0f;
    Quad quad;

    std::size_t bodyStart = 0;
    std::size_t bodyEnd = 0;

    Node() = default;

    inline Node(int next,
                const Quad &quad,
                std::size_t bodyStart,
                std::size_t bodyEnd)
        : children(0),
          next(next),
          pos(Vec2()),
          mass(0.0f),
          quad(quad),
          bodyStart(bodyStart),
          bodyEnd(bodyEnd) {}

    inline bool isLeaf() const {
        return children == 0;
    }

    inline bool isBranch() const {
        return children != 0;
    }

    inline bool isEmpty() const {
        return mass == 0.0f;
    }
};

class QuadTree {
public:
    static constexpr int ROOT = 0;

    QuadTree(float theta, float epsilon, std::size_t leafCapacity)
      : tSq(theta * theta),
        eSq(epsilon * epsilon),
        leafCapacity(leafCapacity) {}

    void build(std::vector<Body> &bodies);

    Vec2 acceleration(
        const Vec2 &pos,
        const std::vector<Body> &bodies) const;

private:
    void subdivide(int node,
                   int children,
                   std::vector<Body> &bodies,
                   std::size_t start,
                   std::size_t end);

    void accumulateLeaf(int node, const std::vector<Body> &bodies);

    // Subdivides `node` into the 4 slots at `children`, or makes it a leaf
    // when `children` is 0.
    void processNode(int node, int children, std::vector<Body> &bodies);

    void propagate();

    float tSq;
    float eSq;
    std::size_t leafCapacity;
    std::vector<Node> nodes;
    std::vector<int> parents;
    std::vector<int> childSlot; // per-level scratch for build()
};