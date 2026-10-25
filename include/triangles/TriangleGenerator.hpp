#pragma once

#include <cstddef>
#include <vector>

#include "triangles/Vertex.hpp"

namespace triangles {

// Creates triangleCount randomly positioned and sized triangles.
// The returned flat vertex list is suitable for VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST.
// triangleCount must be in the inclusive range [1, 100000].
std::vector<Vertex> generateTriangleVertices(std::size_t triangleCount);

} // namespace triangles
