#pragma once
#include <cmath>

struct Vec2 {
  float x, y;

  Vec2() : x(0.0f), y(0.0f) {}
  Vec2(float x, float y) : x(x), y(y) {}

  Vec2 operator+(const Vec2 &other) const { return {x + other.x, y + other.y}; }

  Vec2 operator-(const Vec2 &other) const { return {x - other.x, y - other.y}; }

  Vec2 operator*(float scalar) const { return {x * scalar, y * scalar}; }

  Vec2 operator/(float scalar) const { return {x / scalar, y / scalar}; }

  Vec2 &operator+=(const Vec2 &other) {
    x += other.x;
    y += other.y;
    return *this;
  }

  Vec2 &operator-=(const Vec2 &other) {
    x -= other.x;
    y -= other.y;
    return *this;
  }

  Vec2 &operator*=(float scalar) {
    x *= scalar;
    y *= scalar;
    return *this;
  }

  Vec2 &operator/=(float scalar) {
    x /= scalar;
    y /= scalar;
    return *this;
  }

  bool operator==(const Vec2 &other) const {
    return x == other.x && y == other.y;
  }

  bool operator!=(const Vec2 &other) const { return !(*this == other); }

  float magSq() const { return x * x + y * y; }
  float mag() const { return std::sqrt(magSq()); }
  float dot(const Vec2 &other) const { return x * other.x + y * other.y; }
  static Vec2 zero() { return Vec2(0.0f, 0.0f); }
  static Vec2 one() { return Vec2(1.0f, 1.0f); }
};
inline Vec2 operator*(float scalar, const Vec2 &v) { return v * scalar; }
