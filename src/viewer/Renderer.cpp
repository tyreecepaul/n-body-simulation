#include "Renderer.h"

#include <GL/glew.h>

#include <cstdio>

namespace {
const char *kVertexShader = R"(#version 430 core
layout(location = 0) in vec4 aBody; // x, y, speed, radius
uniform vec2 uCenter;
uniform float uScale;
uniform vec2 uViewport;
uniform float uPointScale;
uniform float uMinPoint;
uniform float uSpeedMax;
uniform int uMode;
out vec3 vColor;
out float vAlpha;

vec3 ramp(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 slow = vec3(0.15, 0.3, 1.0), mid = vec3(0.2, 0.95, 1.0),
       fast = vec3(1.0, 0.9, 0.55);
  return t < 0.5 ? mix(slow, mid, t * 2.0) : mix(mid, fast, t * 2.0 - 1.0);
}

void main() {
  vec2 p = (aBody.xy - uCenter) * uScale;
  gl_Position = vec4(p * 2.0 / uViewport, 0.0, 1.0);
  float size = aBody.w * 2.0 * uScale * uPointScale;
  gl_PointSize = max(size, uMinPoint);
  // Bodies smaller than the minimum point fade instead of all looking
  // equally bright, so density still reads when zoomed out.
  vAlpha = clamp(size / uMinPoint, 0.08, 1.0);
  vColor = uMode == 0 ? ramp(aBody.z / uSpeedMax) : vec3(0.35, 0.75, 1.0);
}
)";

const char *kFragmentShader = R"(#version 430 core
in vec3 vColor;
in float vAlpha;
uniform float uBrightness;
out vec4 fragColor;

void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  float a = (1.0 - r2) * vAlpha * uBrightness;
  fragColor = vec4(vColor * a, a); // premultiplied
}
)";

GLuint compile(GLenum type, const char *source) {
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint ok = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    std::fprintf(stderr, "Shader compile error:\n%s\n", log);
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}
} // namespace

bool Renderer::init() {
  const GLuint vs = compile(GL_VERTEX_SHADER, kVertexShader);
  const GLuint fs = compile(GL_FRAGMENT_SHADER, kFragmentShader);
  if (!vs || !fs) {
    return false;
  }
  program_ = glCreateProgram();
  glAttachShader(program_, vs);
  glAttachShader(program_, fs);
  glLinkProgram(program_);
  glDeleteShader(vs);
  glDeleteShader(fs);
  GLint ok = 0;
  glGetProgramiv(program_, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetProgramInfoLog(program_, sizeof(log), nullptr, log);
    std::fprintf(stderr, "Shader link error:\n%s\n", log);
    return false;
  }
  uCenter_ = glGetUniformLocation(program_, "uCenter");
  uScale_ = glGetUniformLocation(program_, "uScale");
  uViewport_ = glGetUniformLocation(program_, "uViewport");
  uPointScale_ = glGetUniformLocation(program_, "uPointScale");
  uMinPoint_ = glGetUniformLocation(program_, "uMinPoint");
  uSpeedMax_ = glGetUniformLocation(program_, "uSpeedMax");
  uMode_ = glGetUniformLocation(program_, "uMode");
  uBrightness_ = glGetUniformLocation(program_, "uBrightness");
  glGenVertexArrays(1, &vao_);
  return true;
}

Renderer::~Renderer() {
  if (vao_) {
    glDeleteVertexArrays(1, &vao_);
  }
  if (program_) {
    glDeleteProgram(program_);
  }
}

void Renderer::draw(unsigned vbo, std::size_t count, const Camera &camera,
                    int widthPx, int heightPx, const DrawSettings &s) {
  glViewport(0, 0, widthPx, heightPx);
  glClearColor(0.0f, 0.0f, 0.02f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  if (count == 0) {
    return;
  }
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_BLEND);
  // Glow adds light where bodies overlap; Speed composites them normally.
  if (s.mode == ColorMode::Glow) {
    glBlendFunc(GL_ONE, GL_ONE);
  } else {
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  }
  glUseProgram(program_);
  glUniform2f(uCenter_, camera.centerX, camera.centerY);
  glUniform1f(uScale_, camera.scale);
  glUniform2f(uViewport_, static_cast<float>(widthPx),
              static_cast<float>(heightPx));
  glUniform1f(uPointScale_, s.pointScale);
  glUniform1f(uMinPoint_, s.minPointPx);
  glUniform1f(uSpeedMax_, s.speedMax);
  glUniform1i(uMode_, s.mode == ColorMode::Speed ? 0 : 1);
  glUniform1f(uBrightness_, s.brightness);

  glBindVertexArray(vao_);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
  glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(count));
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glUseProgram(0);
  glDisable(GL_BLEND);
}
