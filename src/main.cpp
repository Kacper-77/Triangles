#include "triangles/Renderer.hpp"
#include "triangles/TriangleGenerator.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr int DEFAULT_TRIANGLE_COUNT = 1000;
constexpr int MIN_TRIANGLE_COUNT = 1;
constexpr int MAX_TRIANGLE_COUNT = 100000;

int parseTriangleCount(int argc, char* argv[]) {
    if (argc > 2) {
        throw std::invalid_argument("Usage: Triangles [triangle_count] (1-100000)");
    }
    if (argc == 1) return DEFAULT_TRIANGLE_COUNT;

    try {
        const std::string argument = argv[1];
        std::size_t parsedCharacters = 0;
        const long long value = std::stoll(argument, &parsedCharacters);
        if (parsedCharacters != argument.size() || value < MIN_TRIANGLE_COUNT ||
            value > MAX_TRIANGLE_COUNT) {
            throw std::invalid_argument("out of range");
        }
        return static_cast<int>(value);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid triangle count. Usage: Triangles [triangle_count] (1-100000)");
    }
}
} // namespace

int main(int argc, char* argv[]) {
    try {
        const int triangleCount = parseTriangleCount(argc, argv);
        const auto vertices = triangles::generateTriangleVertices(
            static_cast<std::size_t>(triangleCount));
        std::cout << "Rendering " << triangleCount << " randomly generated triangles ("
                  << vertices.size() << " vertices).\n";
        triangles::Renderer renderer;
        renderer.run(vertices);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
