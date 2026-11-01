#include "Snapshot.h"

#include <cstdint>
#include <cstring>
#include <fstream>

namespace {
// 28-byte header, as in src/Main.cpp.
struct Header {
  char magic[4];
  std::uint32_t nBodies, totalSteps, outputEvery, numFrames, completedFrames;
  float dt;
};
static_assert(sizeof(Header) == 28, "NBS2 header must be 28 bytes");
} // namespace

bool saveSnapshot(const std::string &path, const std::vector<Body> &bodies,
                  float dt, std::string &error) {
  const std::size_t n = bodies.size();
  std::vector<float> mass(n), radius(n), frame(n * 4);
  std::vector<bool> seen(n, false);
  for (const Body &b : bodies) {
    if (b.id >= n || seen[b.id]) {
      error = "body ids are not 0..n-1";
      return false;
    }
    seen[b.id] = true;
    mass[b.id] = b.mass;
    radius[b.id] = b.radius;
    float *record = &frame[static_cast<std::size_t>(b.id) * 4];
    record[0] = b.pos.x;
    record[1] = b.pos.y;
    record[2] = b.vel.x;
    record[3] = b.vel.y;
  }

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    error = "cannot open " + path;
    return false;
  }
  Header h{{'N', 'B', 'S', '2'}, static_cast<std::uint32_t>(n), 0, 1, 1, 1,
           dt};
  out.write(reinterpret_cast<const char *>(&h), sizeof(h));
  out.write(reinterpret_cast<const char *>(mass.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
  out.write(reinterpret_cast<const char *>(radius.data()),
            static_cast<std::streamsize>(n * sizeof(float)));
  out.write(reinterpret_cast<const char *>(frame.data()),
            static_cast<std::streamsize>(frame.size() * sizeof(float)));
  out.close();
  if (!out) {
    error = "error while writing " + path;
    return false;
  }
  return true;
}

bool loadSnapshot(const std::string &path, std::vector<Body> &bodies,
                  float &dt, std::string &error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open " + path;
    return false;
  }
  Header h{};
  in.read(reinterpret_cast<char *>(&h), sizeof(h));
  if (!in || std::memcmp(h.magic, "NBS2", 4) != 0) {
    error = path + " is not an NBS2 file";
    return false;
  }
  if (h.completedFrames == 0) {
    error = path + " has no completed frames";
    return false;
  }
  const std::size_t n = h.nBodies;
  std::vector<float> mass(n), radius(n), frame(n * 4);
  in.read(reinterpret_cast<char *>(mass.data()),
          static_cast<std::streamsize>(n * sizeof(float)));
  in.read(reinterpret_cast<char *>(radius.data()),
          static_cast<std::streamsize>(n * sizeof(float)));
  const std::streamoff frameBytes =
      static_cast<std::streamoff>(n * 4 * sizeof(float));
  in.seekg(static_cast<std::streamoff>(h.completedFrames - 1) * frameBytes,
           std::ios::cur);
  in.read(reinterpret_cast<char *>(frame.data()), frameBytes);
  if (!in) {
    error = path + " is truncated";
    return false;
  }

  bodies.clear();
  bodies.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float *r = &frame[i * 4];
    bodies.emplace_back(Vec2(r[0], r[1]), Vec2(r[2], r[3]), mass[i],
                        radius[i], static_cast<std::uint32_t>(i));
  }
  dt = h.dt;
  return true;
}
