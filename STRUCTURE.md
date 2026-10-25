# Structure

- `src/main.cpp` — CLI and application startup
- `src/TriangleGenerator.cpp` — random triangle geometry
- `src/Renderer.cpp` — Vulkan renderer, camera, depth testing, FPS and optional frustum culling
- `include/triangles/Vertex.hpp` — shared vertex layout
- `include/triangles/TriangleGenerator.hpp` — generator API
- `include/triangles/Renderer.hpp` — renderer API
- `shaders/` — GLSL shaders compiled to SPIR-V by CMake
