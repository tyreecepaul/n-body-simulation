#include "CollisionKernels.cuh"
#include "CudaCollision.cuh"

#include <cub/cub.cuh>

#include <algorithm>

namespace {
using namespace collision_detail;

constexpr int kBlock = 256;
constexpr int kLargeBlock = 128; // power of two, for the block reduction
constexpr int kLargeChunk = 4096; // bodies per block in the split large scan

// Grid key per body (kLargeKey for large boxes) and identity values for
// the sort; counts large boxes (integer atomic, so deterministic).
__global__ void keyKernel(int n, float cell, float invCell,
                          const float *__restrict__ posX,
                          const float *__restrict__ posY,
                          const float *__restrict__ radius,
                          unsigned long long *__restrict__ keys,
                          int *__restrict__ index, int *largeCount) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const unsigned long long key =
      boxKey(makeBox(i, posX, posY, radius), cell, invCell);
  keys[i] = key;
  index[i] = i;
  if (key == kLargeKey) {
    atomicAdd(largeCount, 1);
  }
}

// One thread per small body, in sorted (cell) order so neighbouring
// threads touch the same cells. Scans the full 3x3 stencil of small boxes
// plus every large box, in a fixed order.
__global__ void smallBodyKernel(
    int n, const int *__restrict__ largeCount, float invCell,
    const unsigned long long *__restrict__ keysSorted,
    const int *__restrict__ indexSorted, const float *__restrict__ posX,
    const float *__restrict__ posY, const float *__restrict__ velX,
    const float *__restrict__ velY, const float *__restrict__ mass,
    const float *__restrict__ radius, float *__restrict__ outPosX,
    float *__restrict__ outPosY, float *__restrict__ outVelX,
    float *__restrict__ outVelY, int *__restrict__ contacts,
    int *__restrict__ aabbAsLater) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  const int nSmall = n - *largeCount;
  if (s >= nSmall) {
    return;
  }
  const BodyArrays bodies{posX, posY, velX, velY, mass, radius};
  const int a = indexSorted[s];
  const Accum acc = smallBodyContacts(s, n, nSmall, invCell, keysSorted,
                                      indexSorted, bodies);
  writeResult(a, loadState(a, bodies), acc, outPosX, outPosY, outVelX,
              outVelY, contacts, aabbAsLater);
}

// Fixed-shape tree sum over the block (deterministic for a fixed
// kLargeBlock). All threads get the total.
template <typename T> __device__ T blockSum(T value, T *shared) {
  const int tid = static_cast<int>(threadIdx.x);
  shared[tid] = value;
  __syncthreads();
  for (int stride = kLargeBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const T total = shared[0];
  __syncthreads();
  return total;
}

// Block-wide sum of every Accum field (fixed reduction shape).
__device__ Accum blockSumAccum(const Accum &mine, float *sharedF,
                               int *sharedI) {
  Accum total;
  total.dpx = blockSum(mine.dpx, sharedF);
  total.dpy = blockSum(mine.dpy, sharedF);
  total.dvx = blockSum(mine.dvx, sharedF);
  total.dvy = blockSum(mine.dvy, sharedF);
  total.npx = blockSum(mine.npx, sharedF);
  total.npy = blockSum(mine.npy, sharedF);
  total.nvx = blockSum(mine.nvx, sharedF);
  total.nvy = blockSum(mine.nvy, sharedF);
  total.contacts = blockSum(mine.contacts, sharedI);
  total.aabbAsLater = blockSum(mine.aabbAsLater, sharedI);
  return total;
}

// Reference path: one block per large body, its threads striding over all
// n bodies. Simple but uses one SM per large body (slow at large n).
__global__ void largeBodyKernel(
    int n, const int *__restrict__ largeCount,
    const int *__restrict__ indexSorted, const float *__restrict__ posX,
    const float *__restrict__ posY, const float *__restrict__ velX,
    const float *__restrict__ velY, const float *__restrict__ mass,
    const float *__restrict__ radius, float *__restrict__ outPosX,
    float *__restrict__ outPosY, float *__restrict__ outVelX,
    float *__restrict__ outVelY, int *__restrict__ contacts,
    int *__restrict__ aabbAsLater) {
  __shared__ float sharedF[kLargeBlock];
  __shared__ int sharedI[kLargeBlock];

  const BodyArrays bodies{posX, posY, velX, velY, mass, radius};
  const int nSmall = n - *largeCount;
  const int a = indexSorted[nSmall + blockIdx.x];
  const Box boxA = makeBox(a, posX, posY, radius);
  const BodyState sa = loadState(a, bodies);

  Accum mine;
  for (int b = threadIdx.x; b < n; b += kLargeBlock) {
    if (b != a) {
      accumulatePair(boxA, sa, b, bodies, mine);
    }
  }

  const Accum total = blockSumAccum(mine, sharedF, sharedI);
  if (threadIdx.x == 0) {
    writeResult(a, sa, total, outPosX, outPosY, outVelX, outVelY, contacts,
                aabbAsLater);
  }
}

// Default path, step 1: grid (chunks, largeCount). Block (c, l) scans
// bodies [c * kLargeChunk, (c + 1) * kLargeChunk) against large body l and
// stores the block's combined Accum as one partial.
__global__ void largeBodyPartialKernel(
    int n, int chunks, const int *__restrict__ largeCount,
    const int *__restrict__ indexSorted, const float *__restrict__ posX,
    const float *__restrict__ posY, const float *__restrict__ velX,
    const float *__restrict__ velY, const float *__restrict__ mass,
    const float *__restrict__ radius, Accum *__restrict__ partials) {
  __shared__ float sharedF[kLargeBlock];
  __shared__ int sharedI[kLargeBlock];

  const BodyArrays bodies{posX, posY, velX, velY, mass, radius};
  const int nSmall = n - *largeCount;
  const int large = static_cast<int>(blockIdx.y);
  const int a = indexSorted[nSmall + large];
  const Box boxA = makeBox(a, posX, posY, radius);
  const BodyState sa = loadState(a, bodies);

  const int begin = static_cast<int>(blockIdx.x) * kLargeChunk;
  const int end = min(n, begin + kLargeChunk);
  Accum mine;
  for (int b = begin + static_cast<int>(threadIdx.x); b < end;
       b += kLargeBlock) {
    if (b != a) {
      accumulatePair(boxA, sa, b, bodies, mine);
    }
  }
  const Accum total = blockSumAccum(mine, sharedF, sharedI);
  if (threadIdx.x == 0) {
    partials[large * chunks + static_cast<int>(blockIdx.x)] = total;
  }
}

// Default path, step 2: one thread per large body sums its partials in
// chunk order (fixed, so deterministic) and writes the result.
__global__ void largeBodyCombineKernel(
    int n, int chunks, const int *__restrict__ largeCount,
    const int *__restrict__ indexSorted, const float *__restrict__ posX,
    const float *__restrict__ posY, const float *__restrict__ velX,
    const float *__restrict__ velY, const float *__restrict__ mass,
    const float *__restrict__ radius, const Accum *__restrict__ partials,
    float *__restrict__ outPosX, float *__restrict__ outPosY,
    float *__restrict__ outVelX, float *__restrict__ outVelY,
    int *__restrict__ contacts, int *__restrict__ aabbAsLater) {
  const int large = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int nLarge = *largeCount;
  if (large >= nLarge) {
    return;
  }
  const BodyArrays bodies{posX, posY, velX, velY, mass, radius};
  const int a = indexSorted[n - nLarge + large];
  Accum total;
  for (int c = 0; c < chunks; ++c) {
    const Accum &p = partials[large * chunks + c];
    total.dpx += p.dpx;
    total.dpy += p.dpy;
    total.dvx += p.dvx;
    total.dvy += p.dvy;
    total.npx += p.npx;
    total.npy += p.npy;
    total.nvx += p.nvx;
    total.nvy += p.nvy;
    total.contacts += p.contacts;
    total.aabbAsLater += p.aabbAsLater;
  }
  writeResult(a, loadState(a, bodies), total, outPosX, outPosY, outVelX,
              outVelY, contacts, aabbAsLater);
}
} // namespace

CudaCollider::CudaCollider(const DeviceBodies &b) : n_(b.n) {
  // cell = 1.01 x median box width (box width = 2 x radius), as in
  // Collision.cpp; the 1% slack absorbs rounding in pos +- radius.
  std::vector<float> radii;
  b.radius.download(radii);
  if (!radii.empty()) {
    std::nth_element(radii.begin(), radii.begin() + radii.size() / 2,
                     radii.end());
    cell_ = 2.0f * radii[radii.size() / 2] * 1.01f;
  }
  if (!(cell_ > 0.0f)) {
    cell_ = 1.0f;
  }

  keys_.resize(n_);
  keysSorted_.resize(n_);
  index_.resize(n_);
  indexSorted_.resize(n_);
  largeCount_.resize(1);
  outPosX_.resize(n_);
  outPosY_.resize(n_);
  outVelX_.resize(n_);
  outVelY_.resize(n_);
  contacts_.resize(n_);
  aabbAsLater_.resize(n_);

  // Ask CUB how much scratch the sort needs (null temp pointer = query).
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, sortTempBytes_, keys_.data(), keysSorted_.data(),
      index_.data(), indexSorted_.data(), static_cast<int>(n_)));
  sortTemp_.resize(std::max<std::size_t>(sortTempBytes_, 1));
}

void CudaCollider::collide(DeviceBodies &b) {
  const int n = static_cast<int>(n_);
  if (n < 2) {
    return;
  }
  const float invCell = 1.0f / cell_;
  const int blocks = (n + kBlock - 1) / kBlock;

  CUDA_CHECK(cudaMemset(largeCount_.data(), 0, sizeof(int)));
  keyKernel<<<blocks, kBlock>>>(n, cell_, invCell, b.posX.data(),
                                b.posY.data(), b.radius.data(), keys_.data(),
                                index_.data(), largeCount_.data());
  CUDA_CHECK_LAUNCH();

  // Stable radix sort: equal keys keep index order, so the scan order (and
  // therefore every float sum) is the same on every run.
  std::size_t tempBytes = sortTempBytes_;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      sortTemp_.data(), tempBytes, keys_.data(), keysSorted_.data(),
      index_.data(), indexSorted_.data(), n));

  smallBodyKernel<<<blocks, kBlock>>>(
      n, largeCount_.data(), invCell, keysSorted_.data(), indexSorted_.data(),
      b.posX.data(), b.posY.data(), b.velX.data(), b.velY.data(),
      b.mass.data(), b.radius.data(), outPosX_.data(), outPosY_.data(),
      outVelX_.data(), outVelY_.data(), contacts_.data(), aabbAsLater_.data());
  CUDA_CHECK_LAUNCH();

  // The large-body grid size is needed on the host: one 4-byte copy.
  int largeCount = 0;
  CUDA_CHECK(cudaMemcpy(&largeCount, largeCount_.data(), sizeof(int),
                        cudaMemcpyDeviceToHost));
  if (largeCount > 0 && splitLargeScan_) {
    // Spread each large body's scan over many blocks, then combine.
    const int chunks = (n + kLargeChunk - 1) / kLargeChunk;
    const std::size_t bytes =
        static_cast<std::size_t>(largeCount) * chunks * sizeof(Accum);
    if (largePartials_.size() < bytes) {
      largePartials_.resize(bytes);
    }
    Accum *partials = reinterpret_cast<Accum *>(largePartials_.data());
    largeBodyPartialKernel<<<dim3(chunks, largeCount), kLargeBlock>>>(
        n, chunks, largeCount_.data(), indexSorted_.data(), b.posX.data(),
        b.posY.data(), b.velX.data(), b.velY.data(), b.mass.data(),
        b.radius.data(), partials);
    CUDA_CHECK_LAUNCH();
    largeBodyCombineKernel<<<(largeCount + kBlock - 1) / kBlock, kBlock>>>(
        n, chunks, largeCount_.data(), indexSorted_.data(), b.posX.data(),
        b.posY.data(), b.velX.data(), b.velY.data(), b.mass.data(),
        b.radius.data(), partials, outPosX_.data(), outPosY_.data(),
        outVelX_.data(), outVelY_.data(), contacts_.data(),
        aabbAsLater_.data());
    CUDA_CHECK_LAUNCH();
  } else if (largeCount > 0) {
    largeBodyKernel<<<largeCount, kLargeBlock>>>(
        n, largeCount_.data(), indexSorted_.data(), b.posX.data(),
        b.posY.data(), b.velX.data(), b.velY.data(), b.mass.data(),
        b.radius.data(), outPosX_.data(), outPosY_.data(), outVelX_.data(),
        outVelY_.data(), contacts_.data(), aabbAsLater_.data());
    CUDA_CHECK_LAUNCH();
  }

  // Every body's output was written by exactly one of the two kernels;
  // make the outputs the current state.
  b.posX.swap(outPosX_);
  b.posY.swap(outPosY_);
  b.velX.swap(outVelX_);
  b.velY.swap(outVelY_);
}

std::int64_t CudaCollider::lastAabbPairs() const {
  std::vector<int> perBody;
  aabbAsLater_.download(perBody);
  std::int64_t total = 0;
  for (int c : perBody) {
    total += c;
  }
  return total;
}

void CudaCollider::lastContacts(std::vector<int> &contacts) const {
  contacts_.download(contacts);
}
