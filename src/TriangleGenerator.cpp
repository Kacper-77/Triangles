#include "triangles/TriangleGenerator.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace triangles {

std::vector<Vertex> generateTriangleVertices(std::size_t triangleCount) {
    constexpr std::size_t minTriangleCount = 1;
    constexpr std::size_t maxTriangleCount = 100000;

    if (triangleCount < minTriangleCount ||
        triangleCount > maxTriangleCount) {
        throw std::invalid_argument(
            "Triangle count must be between 1 and 100000.");
    }

    std::random_device randomDevice;
    std::mt19937 randomEngine(randomDevice());

    std::uniform_real_distribution<float> center(-4.5f, 4.5f);
    std::uniform_real_distribution<float> size(0.15f, 1.0f);
    std::uniform_real_distribution<float> depth(-3.5f, 3.5f);
    std::uniform_real_distribution<float> angle(
        0.0f, 6.28318530718f);
    std::uniform_real_distribution<float> color(0.2f, 1.0f);

    std::vector<Vertex> vertices;
    vertices.reserve(triangleCount * 3);

    for (std::size_t i = 0; i < triangleCount; ++i) {
        const float x = center(randomEngine);
        const float y = center(randomEngine);
        const float z = depth(randomEngine);
        const float radius = size(randomEngine);
        const float rotation = angle(randomEngine);

        const float red = color(randomEngine);
        const float green = color(randomEngine);
        const float blue = color(randomEngine);

        const float cosA = std::cos(rotation);
        const float sinA = std::sin(rotation);

        // Equilateral triangle in local XY plane.
        const float localX[3] = {
            0.0f, -0.8660254f, 0.8660254f
        };
        const float localY[3] = {
            1.0f, -0.5f, -0.5f
        };

        for (int v = 0; v < 3; ++v) {
            const float px = localX[v] * radius;
            const float py = localY[v] * radius;

            const float rotatedX = px * cosA - py * sinA;
            const float rotatedY = px * sinA + py * cosA;

            vertices.push_back({
                {x + rotatedX, y + rotatedY, z},
                {red, green, blue}
            });
        }
    }

    return vertices;
}

} // namespace triangles
