# GPU build: the CUDA simulation programs and the real-time viewer.
#   cmake -B build-cmake && cmake --build build-cmake -j
# (The CPU programs and the cluster scripts use the Makefile.)
#
# Host compiler too new for nvcc?  -DCMAKE_CUDA_HOST_COMPILER=g++-15
cmake_minimum_required(VERSION 3.24)
project(nbody LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

set(SRC ${CMAKE_SOURCE_DIR}/src)

# CPU code, built as the Makefile's CUDA rules do: optimised, no OpenMP
# (the pragmas are ignored).
set(CPU_FLAGS -Ofast -ffast-math -march=native -Wall -Wextra -Wno-unknown-pragmas)

add_library(nbody_common STATIC ${SRC}/Utils.cpp)
target_include_directories(nbody_common PUBLIC ${SRC})
target_compile_options(nbody_common PRIVATE ${CPU_FLAGS})

# CPU reference implementation, for cuda_check.
add_library(nbody_cpu_ref STATIC
  ${SRC}/Collision.cpp ${SRC}/Simulation.cpp ${SRC}/QuadTree.cpp
  ${SRC}/naive/NaiveSimulation.cpp)
target_include_directories(nbody_cpu_ref PUBLIC ${SRC} ${SRC}/naive)
target_compile_options(nbody_cpu_ref PRIVATE ${CPU_FLAGS})
target_link_libraries(nbody_cpu_ref PUBLIC nbody_common)

add_library(nbody_cuda STATIC
  ${SRC}/cuda/DeviceBodies.cu ${SRC}/cuda/CudaCollision.cu
  ${SRC}/cuda/CudaNaive.cu ${SRC}/cuda/CudaQuadTree.cu
  ${SRC}/cuda/CudaSimulation.cu)
target_include_directories(nbody_cuda PUBLIC ${SRC} ${SRC}/cuda ${SRC}/naive)
target_compile_options(nbody_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xcompiler=-O3>)
target_link_libraries(nbody_cuda PUBLIC nbody_common CUDA::cudart)

add_executable(sim_cuda ${SRC}/cuda/MainCuda.cu)
target_link_libraries(sim_cuda PRIVATE nbody_cuda)

add_executable(naive_sim_cuda ${SRC}/cuda/MainNaiveCuda.cu)
target_link_libraries(naive_sim_cuda PRIVATE nbody_cuda)

add_executable(cuda_check ${SRC}/cuda/CheckCuda.cu)
target_link_libraries(cuda_check PRIVATE nbody_cuda nbody_cpu_ref)

# ---------------- Viewer ----------------
option(NBODY_VIEWER "Build the real-time viewer (needs SDL3, GLEW, OpenGL)" ON)
if(NBODY_VIEWER)
  find_package(SDL3 REQUIRED CONFIG)
  find_package(GLEW REQUIRED)
  find_package(OpenGL REQUIRED)

  include(FetchContent)
  FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.9b
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(imgui)

  add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
  target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends)
  target_link_libraries(imgui PUBLIC SDL3::SDL3 OpenGL::GL)

  add_executable(nbody_viewer
    ${SRC}/viewer/main.cpp ${SRC}/viewer/Interop.cu ${SRC}/viewer/Renderer.cpp
    ${SRC}/viewer/Recorder.cpp ${SRC}/viewer/Scenarios.cpp
    ${SRC}/viewer/Snapshot.cpp)
  target_include_directories(nbody_viewer PRIVATE ${SRC}/viewer)
  target_compile_options(nbody_viewer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra>)
  target_link_libraries(nbody_viewer PRIVATE nbody_cuda imgui GLEW::GLEW
    OpenGL::GL SDL3::SDL3)
endif()
