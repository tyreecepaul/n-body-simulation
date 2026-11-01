// nbody_viewer: real-time GPU N-body simulation. The CUDA Barnes-Hut
// simulation (src/cuda/) steps on the GPU and is drawn straight from GPU
// memory through CUDA-OpenGL interop, with an ImGui control panel.
//
// Controls (when the mouse isn't over the panel):
//   left drag            pan              wheel          zoom at cursor
//   right hold           attractor        shift + click  drop heavy body
//   ctrl + drag          launch a cluster along the drag
//   space pause   .  single step   F follow   R record   H hide panel   Esc quit

#include <GL/glew.h>
#include <SDL3/SDL.h>

#include "CudaSimulation.cuh"
#include "Interop.cuh"
#include "Recorder.h"
#include "Renderer.h"
#include "Scenarios.h"
#include "Snapshot.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl3.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Viewer {
  // Simulation
  std::unique_ptr<CudaSimulation> sim;
  GlBodyBuffer buffer;
  Scenario scenario = Scenario::Disc;
  int bodyCount = 1'000'000;
  unsigned seed = 0;

  // Run state
  bool paused = false;
  bool stepOnce = false;
  int stepsPerFrame = 1;
  bool autoSteps = false;
  float msPerStep = 0.0f;
  int nodes = 0, levels = 0;

  // View
  Renderer renderer;
  Camera camera;
  DrawSettings draw;
  bool autoSpeedScale = true;
  bool follow = false;
  bool showPanel = true;
  int widthPx = 1600, heightPx = 1000; // drawable size
  float pixelRatio = 1.0f;              // drawable pixels per window unit

  // Interaction
  float heavyMass = 2e5f;
  int clusterSize = 5000;
  float launchScale = 0.05f; // launch velocity per world unit dragged
  float mouseX = 0.0f, mouseY = 0.0f; // drawable pixels
  bool panning = false;
  bool launching = false;
  float launchStartX = 0.0f, launchStartY = 0.0f; // world
  bool attracting = false;

  // Files
  Recorder recorder;
  bool recordPanel = false;
  char snapshotPath[256] = "snapshots/snapshot.bin";
  std::string message;

  float worldX(float px) const {
    return camera.centerX + (px - widthPx * 0.5f) / camera.scale;
  }
  float worldY(float py) const {
    return camera.centerY + (heightPx * 0.5f - py) / camera.scale;
  }

  void fitCamera(const std::vector<Body> &bodies) {
    float minX = 1e30f, maxX = -1e30f, minY = 1e30f, maxY = -1e30f;
    for (const Body &b : bodies) {
      minX = std::min(minX, b.pos.x);
      maxX = std::max(maxX, b.pos.x);
      minY = std::min(minY, b.pos.y);
      maxY = std::max(maxY, b.pos.y);
    }
    if (bodies.empty()) {
      return;
    }
    camera.centerX = 0.5f * (minX + maxX);
    camera.centerY = 0.5f * (minY + maxY);
    const float span = std::max({maxX - minX, maxY - minY, 1.0f}) * 1.1f;
    camera.scale = std::min(widthPx, heightPx) / span;
  }

  // Replaces the whole state, keeping the simulation parameters.
  void load(const std::vector<Body> &bodies, bool refit) {
    if (sim) {
      sim->reset(bodies);
    } else {
      sim = std::make_unique<CudaSimulation>(bodies);
    }
    buffer.resize(bodies.size());
    if (refit) {
      fitCamera(bodies);
    }
  }

  void resetScenario() {
    message.clear();
    load(makeScenario(scenario, static_cast<std::size_t>(bodyCount), seed),
         true);
  }

  // Copies the state back from the GPU, edits it, and uploads it again.
  template <typename Edit> void editBodies(Edit edit) {
    std::vector<Body> bodies;
    sim->bodies.download(bodies);
    edit(bodies);
    renumber(bodies);
    load(bodies, false);
  }

  void stepSimulation() {
    if (paused && !stepOnce) {
      return;
    }
    const int steps = stepOnce ? 1 : stepsPerFrame;
    stepOnce = false;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < steps; ++i) {
      sim->step();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    const float perStep = ms / steps;
    msPerStep = msPerStep == 0.0f ? perStep : 0.9f * msPerStep + 0.1f * perStep;
    if (autoSteps) {
      // Keep simulation time within ~10 ms of each frame.
      stepsPerFrame = std::clamp(static_cast<int>(10.0f / msPerStep), 1, 16);
    }
  }

  void toggleRecording() {
    if (recorder.active()) {
      recorder.stop();
      message = "Saved " + recorder.path() + " (" +
                std::to_string(recorder.frames()) + " frames)";
      return;
    }
    std::filesystem::create_directories("recordings");
    char name[64];
    const std::time_t now = std::time(nullptr);
    std::strftime(name, sizeof(name), "recordings/nbody_%Y%m%d_%H%M%S.mp4",
                  std::localtime(&now));
    message = recorder.start(name, widthPx, heightPx, 60)
                  ? std::string("Recording to ") + name
                  : "Couldn't start ffmpeg";
  }

  void saveSnapshotFile() {
    std::vector<Body> bodies;
    sim->bodies.download(bodies);
    const std::filesystem::path path(snapshotPath);
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path());
    }
    std::string error;
    message = saveSnapshot(snapshotPath, bodies, sim->dt, error)
                  ? std::string("Saved ") + snapshotPath
                  : "Save failed: " + error;
  }

  void loadSnapshotFile() {
    std::vector<Body> bodies;
    float dt = 0.0f;
    std::string error;
    if (!loadSnapshot(snapshotPath, bodies, dt, error)) {
      message = "Load failed: " + error;
      return;
    }
    load(bodies, true);
    sim->dt = dt;
    bodyCount = static_cast<int>(bodies.size());
    message = std::string("Loaded ") + snapshotPath;
  }

  void handleEvent(const SDL_Event &e, bool &running) {
    const ImGuiIO &io = ImGui::GetIO();
    switch (e.type) {
    case SDL_EVENT_QUIT:
      running = false;
      break;
    case SDL_EVENT_KEY_DOWN:
      if (io.WantCaptureKeyboard || e.key.repeat) {
        break;
      }
      switch (e.key.key) {
      case SDLK_ESCAPE: running = false; break;
      case SDLK_SPACE: paused = !paused; break;
      case SDLK_PERIOD: stepOnce = true; break;
      case SDLK_F: follow = !follow; break;
      case SDLK_R: toggleRecording(); break;
      case SDLK_H: showPanel = !showPanel; break;
      default: break;
      }
      break;
    case SDL_EVENT_MOUSE_MOTION:
      mouseX = e.motion.x * pixelRatio;
      mouseY = e.motion.y * pixelRatio;
      if (panning) {
        camera.centerX -= e.motion.xrel * pixelRatio / camera.scale;
        camera.centerY += e.motion.yrel * pixelRatio / camera.scale;
      }
      break;
    case SDL_EVENT_MOUSE_WHEEL: {
      if (io.WantCaptureMouse) {
        break;
      }
      // Zoom about the point under the cursor.
      const float wx = worldX(mouseX), wy = worldY(mouseY);
      camera.scale *= std::pow(1.15f, e.wheel.y);
      camera.centerX += wx - worldX(mouseX);
      camera.centerY += wy - worldY(mouseY);
      break;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
      if (io.WantCaptureMouse) {
        break;
      }
      const SDL_Keymod mod = SDL_GetModState();
      if (e.button.button == SDL_BUTTON_LEFT) {
        if (mod & SDL_KMOD_SHIFT) {
          const float x = worldX(mouseX), y = worldY(mouseY);
          editBodies([&](std::vector<Body> &b) {
            addHeavyBody(b, x, y, heavyMass);
          });
        } else if (mod & SDL_KMOD_CTRL) {
          launching = true;
          launchStartX = worldX(mouseX);
          launchStartY = worldY(mouseY);
        } else {
          panning = true;
        }
      } else if (e.button.button == SDL_BUTTON_RIGHT) {
        attracting = true;
      }
      break;
    }
    case SDL_EVENT_MOUSE_BUTTON_UP:
      if (e.button.button == SDL_BUTTON_LEFT) {
        panning = false;
        if (launching) {
          launching = false;
          const float vx = (worldX(mouseX) - launchStartX) * launchScale;
          const float vy = (worldY(mouseY) - launchStartY) * launchScale;
          const float x = launchStartX, y = launchStartY;
          editBodies([&](std::vector<Body> &b) {
            addCluster(b, x, y, vx, vy, static_cast<std::size_t>(clusterSize),
                       seed + static_cast<unsigned>(b.size()));
          });
        }
      } else if (e.button.button == SDL_BUTTON_RIGHT) {
        attracting = false;
      }
      break;
    default:
      break;
    }
  }

  void drawPanel(float fps) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("N-body");

    ImGui::Text("%.0f FPS   %.2f ms/step   x%d steps/frame", fps, msPerStep,
                stepsPerFrame);
    ImGui::Text("%zu bodies   %d nodes   %d levels", sim->bodies.n, nodes,
                levels);
    ImGui::Text("t = %.1f   (step %zu)", sim->frame * sim->dt, sim->frame);

    if (ImGui::CollapsingHeader("Run", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::Button(paused ? "Play" : "Pause")) {
        paused = !paused;
      }
      ImGui::SameLine();
      if (ImGui::Button("Step")) {
        stepOnce = true;
      }
      ImGui::SameLine();
      ImGui::Checkbox("Auto steps", &autoSteps);
      ImGui::BeginDisabled(autoSteps);
      ImGui::SliderInt("Steps/frame", &stepsPerFrame, 1, 16);
      ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Scenario", ImGuiTreeNodeFlags_DefaultOpen)) {
      int s = static_cast<int>(scenario);
      ImGui::Combo("Preset", &s, kScenarioNames, IM_ARRAYSIZE(kScenarioNames));
      scenario = static_cast<Scenario>(s);
      ImGui::InputInt("Bodies", &bodyCount, 10000, 100000);
      bodyCount = std::clamp(bodyCount, 2, 4'000'000);
      int seedInt = static_cast<int>(seed);
      ImGui::InputInt("Seed", &seedInt);
      seed = static_cast<unsigned>(std::max(seedInt, 0));
      if (ImGui::Button("Reset")) {
        resetScenario();
      }
    }

    if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::SliderFloat("dt", &sim->dt, 0.001f, 0.2f, "%.3f",
                         ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat("theta", &sim->theta, 0.2f, 2.0f, "%.2f");
      ImGui::SliderFloat("epsilon", &sim->epsilon, 0.1f, 20.0f, "%.2f",
                         ImGuiSliderFlags_Logarithmic);
      ImGui::Checkbox("Collisions", &sim->collisions);
    }

    if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
      int mode = static_cast<int>(draw.mode);
      ImGui::Combo("Colour", &mode, "Speed\0Glow\0");
      draw.mode = static_cast<ColorMode>(mode);
      ImGui::SliderFloat("Brightness", &draw.brightness, 0.05f, 2.0f);
      ImGui::SliderFloat("Point scale", &draw.pointScale, 0.1f, 5.0f, "%.2f",
                         ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat("Min point (px)", &draw.minPointPx, 1.0f, 6.0f);
      ImGui::Checkbox("Auto speed range", &autoSpeedScale);
      ImGui::BeginDisabled(autoSpeedScale);
      ImGui::SliderFloat("Max speed", &draw.speedMax, 1.0f, 500.0f, "%.0f",
                         ImGuiSliderFlags_Logarithmic);
      ImGui::EndDisabled();
      ImGui::Checkbox("Follow centre of mass (F)", &follow);
      if (ImGui::Button("Fit view")) {
        std::vector<Body> bodies;
        sim->bodies.download(bodies);
        fitCamera(bodies);
      }
    }

    if (ImGui::CollapsingHeader("Mouse")) {
      ImGui::TextWrapped("Right hold: attractor   Shift+click: heavy body   "
                         "Ctrl+drag: launch cluster");
      ImGui::SliderFloat("Attractor mass", &sim->attractor.mass, 1e4f, 1e8f,
                         "%.2e", ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat("Heavy body mass", &heavyMass, 1e3f, 1e7f, "%.2e",
                         ImGuiSliderFlags_Logarithmic);
      ImGui::SliderInt("Cluster size", &clusterSize, 100, 100000, "%d",
                       ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat("Launch speed", &launchScale, 0.005f, 0.5f, "%.3f",
                         ImGuiSliderFlags_Logarithmic);
    }

    if (ImGui::CollapsingHeader("Record / snapshot")) {
      if (ImGui::Button(recorder.active() ? "Stop recording (R)"
                                          : "Record MP4 (R)")) {
        toggleRecording();
      }
      ImGui::SameLine();
      ImGui::Checkbox("Include panel", &recordPanel);
      ImGui::InputText("File", snapshotPath, sizeof(snapshotPath));
      if (ImGui::Button("Save snapshot")) {
        saveSnapshotFile();
      }
      ImGui::SameLine();
      if (ImGui::Button("Load snapshot")) {
        loadSnapshotFile();
      }
    }

    if (recorder.active()) {
      ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "REC %ld frames",
                         recorder.frames());
    }
    if (!message.empty()) {
      ImGui::TextWrapped("%s", message.c_str());
    }
    ImGui::End();
  }
};

} // namespace

int main(int argc, char **argv) {
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
    return 1;
  }
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_Window *window =
      SDL_CreateWindow("N-body (CUDA Barnes-Hut)", 1600, 1000,
                       SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
  if (!window) {
    std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
    return 1;
  }
  SDL_GLContext context = SDL_GL_CreateContext(window);
  if (!context) {
    std::fprintf(stderr, "OpenGL 4.3 context failed: %s\n", SDL_GetError());
    return 1;
  }
  SDL_GL_MakeCurrent(window, context);
  SDL_GL_SetSwapInterval(1);

  glewExperimental = GL_TRUE;
  const GLenum glewStatus = glewInit();
  // A GLX-built GLEW reports "no GLX display" under Wayland/EGL but has
  // still loaded every function.
  if (glewStatus != GLEW_OK && glewStatus != GLEW_ERROR_NO_GLX_DISPLAY) {
    std::fprintf(stderr, "glewInit failed: %s\n",
                 reinterpret_cast<const char *>(glewGetErrorString(glewStatus)));
    return 1;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr;
  ImGui::StyleColorsDark();
  ImGui_ImplSDL3_InitForOpenGL(window, context);
  ImGui_ImplOpenGL3_Init("#version 430");

  {
    Viewer v;
    if (!v.renderer.init()) {
      return 1;
    }
    SDL_GetWindowSizeInPixels(window, &v.widthPx, &v.heightPx);
    // nbody_viewer [n_bodies | file.bin]
    const std::string arg = argc > 1 ? argv[1] : "";
    if (!arg.empty() && arg.find_first_not_of("0123456789") != arg.npos) {
      std::snprintf(v.snapshotPath, sizeof(v.snapshotPath), "%s",
                    arg.c_str());
      v.loadSnapshotFile();
      if (!v.sim) {
        std::fprintf(stderr, "%s\n", v.message.c_str());
        return 1;
      }
    } else {
      if (!arg.empty()) {
        v.bodyCount = std::stoi(arg);
      }
      v.resetScenario();
    }

    bool running = true;
    auto last = std::chrono::steady_clock::now();
    float fps = 0.0f;
    long frameCounter = 0;
    while (running) {
      SDL_Event e;
      while (SDL_PollEvent(&e)) {
        ImGui_ImplSDL3_ProcessEvent(&e);
        v.handleEvent(e, running);
      }
      int w = 0, h = 0, wPts = 0, hPts = 0;
      SDL_GetWindowSizeInPixels(window, &w, &h);
      SDL_GetWindowSize(window, &wPts, &hPts);
      v.pixelRatio = wPts > 0 ? static_cast<float>(w) / wPts : 1.0f;
      if ((w != v.widthPx || h != v.heightPx) && v.recorder.active()) {
        v.toggleRecording(); // frames must keep one size
      }
      v.widthPx = w;
      v.heightPx = h;

      v.sim->attractor.active = v.attracting;
      v.sim->attractor.x = v.worldX(v.mouseX);
      v.sim->attractor.y = v.worldY(v.mouseY);

      v.stepSimulation();
      v.buffer.fill(v.sim->bodies);
      const BodyStats &stats = v.buffer.stats();
      if (v.follow) {
        v.camera.centerX = stats.comX;
        v.camera.centerY = stats.comY;
      }
      if (v.autoSpeedScale) {
        // Most bodies are far slower than the fastest one.
        v.draw.speedMax =
            0.9f * v.draw.speedMax + 0.1f * (0.5f * stats.maxSpeed);
      }
      if (++frameCounter % 30 == 0) {
        v.nodes = v.sim->tree.nodeCount();
        v.levels = v.sim->tree.levelCount();
      }

      v.renderer.draw(v.buffer.vbo(), v.buffer.size(), v.camera, v.widthPx,
                      v.heightPx, v.draw);
      if (v.recorder.active() && !v.recordPanel) {
        v.recorder.capture();
      }

      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplSDL3_NewFrame();
      ImGui::NewFrame();
      if (v.showPanel) {
        v.drawPanel(fps);
      }
      if (v.launching) {
        // Preview of the launch direction.
        const float r = 1.0f / v.pixelRatio;
        const float sx =
            ((v.launchStartX - v.camera.centerX) * v.camera.scale +
             v.widthPx * 0.5f) * r;
        const float sy =
            (v.heightPx * 0.5f -
             (v.launchStartY - v.camera.centerY) * v.camera.scale) * r;
        ImGui::GetForegroundDrawList()->AddLine(
            ImVec2(sx, sy), ImVec2(v.mouseX * r, v.mouseY * r),
            IM_COL32(255, 200, 80, 255), 2.0f);
      }
      ImGui::Render();
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      if (v.recorder.active() && v.recordPanel) {
        v.recorder.capture();
      }
      SDL_GL_SwapWindow(window);

      const auto now = std::chrono::steady_clock::now();
      const float dt = std::chrono::duration<float>(now - last).count();
      last = now;
      fps = fps == 0.0f ? 1.0f / dt : 0.95f * fps + 0.05f / dt;
    }
    v.recorder.stop();
  } // GL/CUDA objects released while the context still exists

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
  SDL_GL_DestroyContext(context);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
