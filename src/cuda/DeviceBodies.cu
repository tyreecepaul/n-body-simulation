#include "DeviceBodies.cuh"

namespace {
constexpr int kBlock = 256;

__global__ void iterateKernel(int n, float dt, float *__restrict__ posX,
                              float *__restrict__ posY,
                              float *__restrict__ velX,
                              float *__restrict__ velY,
                              const float *__restrict__ accX,
                              const float *__restrict__ accY) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float vx = velX[i] + accX[i] * dt;
  const float vy = velY[i] + accY[i] * dt;
  velX[i] = vx;
  velY[i] = vy;
  posX[i] += vx * dt;
  posY[i] += vy * dt;
}
} // namespace

void DeviceBodies::upload(const std::vector<Body> &bodies) {
  n = bodies.size();
  std::vector<float> px(n), py(n), vx(n), vy(n), ax(n), ay(n), m(n), r(n);
  std::vector<std::uint32_t> ids(n);
  for (std::size_t i = 0; i < n; ++i) {
    const Body &b = bodies[i];
    px[i] = b.pos.x;
    py[i] = b.pos.y;
    vx[i] = b.vel.x;
    vy[i] = b.vel.y;
    ax[i] = b.acc.x;
    ay[i] = b.acc.y;
    m[i] = b.mass;
    r[i] = b.radius;
    ids[i] = b.id;
  }
  posX.upload(px);
  posY.upload(py);
  velX.upload(vx);
  velY.upload(vy);
  accX.upload(ax);
  accY.upload(ay);
  mass.upload(m);
  radius.upload(r);
  id.upload(ids);
}

void DeviceBodies::download(std::vector<Body> &bodies) const {
  std::vector<float> px, py, vx, vy, ax, ay, m, r;
  std::vector<std::uint32_t> ids;
  posX.download(px);
  posY.download(py);
  velX.download(vx);
  velY.download(vy);
  accX.download(ax);
  accY.download(ay);
  mass.download(m);
  radius.download(r);
  id.download(ids);

  bodies.clear();
  bodies.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    Body b(Vec2(px[i], py[i]), Vec2(vx[i], vy[i]), m[i], r[i], ids[i]);
    b.acc = Vec2(ax[i], ay[i]);
    bodies.push_back(b);
  }
}

void iterate(DeviceBodies &b, float dt) {
  const int n = static_cast<int>(b.n);
  if (n == 0) {
    return;
  }
  iterateKernel<<<(n + kBlock - 1) / kBlock, kBlock>>>(
      n, dt, b.posX.data(), b.posY.data(), b.velX.data(), b.velY.data(),
      b.accX.data(), b.accY.data());
  CUDA_CHECK_LAUNCH();
}
