#pragma once

#include <memory>
#include <vector>

#include "triangles/Vertex.hpp"

namespace triangles {

class Renderer {
public:
    Renderer();
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;

    void run(const std::vector<Vertex>& vertices);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace triangles
