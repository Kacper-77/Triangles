#pragma once

namespace triangles {

// Vertex format shared by CPU-side geometry and the future Vulkan vertex buffer.
// Position and color are three 32-bit floats each (24 bytes total on supported targets).
struct Vertex {
    float position[3];
    float color[3];
};

} // namespace triangles
