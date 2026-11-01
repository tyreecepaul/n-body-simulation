#pragma once
// Device-side body storage for the CUDA simulation: structure-of-arrays
// buffers that stay resident on the GPU across steps (bodies only come back
// to the host for output frames and checks).

#include "Body.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

// Aborts with file/line on any CUDA API error.
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t cudaCheckErr_ = (call);                                       \
    if (cudaCheckErr_ != cudaSuccess) {                                       \
      std::fprintf(stderr, "CUDA error: %s\n  at %s:%d\n  in %s\n",           \
                   cudaGetErrorString(cudaCheckErr_), __FILE__, __LINE__,     \
                   #call);                                                    \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// Use right after a kernel launch: reports bad launch configurations
// immediately (execution errors surface at the next synchronising call).
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// Owning, move-only device allocation of n elements of T.
template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) { resize(n); }
  ~DeviceBuffer() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept { swap(other); }
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    swap(other);
    return *this;
  }

  // Reallocates only when the size changes; contents are not preserved.
  void resize(std::size_t n) {
    if (n == size_) {
      return;
    }
    if (ptr_) {
      CUDA_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
    }
    size_ = n;
    if (n > 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr_), n * sizeof(T)));
    }
  }

  void upload(const std::vector<T> &host) {
    resize(host.size());
    if (size_ > 0) {
      CUDA_CHECK(cudaMemcpy(ptr_, host.data(), size_ * sizeof(T),
                            cudaMemcpyHostToDevice));
    }
  }

  void download(std::vector<T> &host) const {
    host.resize(size_);
    if (size_ > 0) {
      CUDA_CHECK(cudaMemcpy(host.data(), ptr_, size_ * sizeof(T),
                            cudaMemcpyDeviceToHost));
    }
  }

  void swap(DeviceBuffer &other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(size_, other.size_);
  }

  T *data() { return ptr_; }
  const T *data() const { return ptr_; }
  std::size_t size() const { return size_; }

private:
  T *ptr_ = nullptr;
  std::size_t size_ = 0;
};

// All per-body state, one device array per field.
struct DeviceBodies {
  std::size_t n = 0;
  DeviceBuffer<float> posX, posY, velX, velY, accX, accY, mass, radius;
  DeviceBuffer<std::uint32_t> id;

  // Copies host bodies (array-of-structs) into the device arrays.
  void upload(const std::vector<Body> &bodies);

  // Overwrites `bodies` (resized to n) with the device state, in device
  // array order (not id order).
  void download(std::vector<Body> &bodies) const;
};

// Semi-implicit Euler, identical to Body::update(): vel += acc*dt, then
// pos += vel*dt.
void iterate(DeviceBodies &b, float dt);
