CXX := g++
CXXFLAGS := -std=c++17 -march=native -Wall -Wextra -Ofast -ffast-math
SRC_DIR := src
BUILD_DIR := build
TARGET := sim

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) nbody_sim_*

# ---------------- CUDA (cluster: module load cuda && make cuda) ----------------
# Builds the GPU binaries from src/cuda/*.cu. cuda_check also links the CPU
# reference code (compiled with $(CXX)) to validate the kernels against it.
# CUDA_BUILD_DIR / CUDA_BIN_DIR can be overridden for job-private builds.
NVCC ?= nvcc
CUDA_ARCH ?= sm_80
NVCCFLAGS := -std=c++17 -O3 -arch=$(CUDA_ARCH) -lineinfo -I$(SRC_DIR) -Xcompiler -O3
CUDA_DIR := $(SRC_DIR)/cuda
CUDA_BUILD_DIR ?= build_cuda
CUDA_BIN_DIR ?= .
CUDA_HEADERS := $(wildcard $(CUDA_DIR)/*.cuh) $(wildcard $(SRC_DIR)/*.h) $(wildcard $(SRC_DIR)/naive/*.h)

CUDA_COMMON := $(CUDA_BUILD_DIR)/DeviceBodies.o $(CUDA_BUILD_DIR)/CudaCollision.o \
               $(CUDA_BUILD_DIR)/CudaNaive.o $(CUDA_BUILD_DIR)/CudaQuadTree.o \
               $(CUDA_BUILD_DIR)/CudaSimulation.o $(CUDA_BUILD_DIR)/cpu/Utils.o
CPU_REF := $(CUDA_BUILD_DIR)/cpu/Collision.o $(CUDA_BUILD_DIR)/cpu/Simulation.o \
           $(CUDA_BUILD_DIR)/cpu/QuadTree.o $(CUDA_BUILD_DIR)/cpu/naive/NaiveSimulation.o
CUDA_TARGETS := $(CUDA_BIN_DIR)/cuda_check $(CUDA_BIN_DIR)/naive_sim_cuda \
                $(CUDA_BIN_DIR)/sim_cuda

.PHONY: cuda clean-cuda

cuda: $(CUDA_TARGETS)

$(CUDA_BIN_DIR)/cuda_check: $(CUDA_BUILD_DIR)/CheckCuda.o $(CUDA_COMMON) $(CPU_REF)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CUDA_BIN_DIR)/naive_sim_cuda: $(CUDA_BUILD_DIR)/MainNaiveCuda.o $(CUDA_COMMON)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CUDA_BIN_DIR)/sim_cuda: $(CUDA_BUILD_DIR)/MainCuda.o $(CUDA_COMMON)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CUDA_BUILD_DIR)/%.o: $(CUDA_DIR)/%.cu $(CUDA_HEADERS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(CUDA_BUILD_DIR)/cpu/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@

clean-cuda:
	rm -rf $(CUDA_BUILD_DIR) $(CUDA_TARGETS)
