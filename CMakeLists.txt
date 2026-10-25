cmake_minimum_required(VERSION 3.20)

project(Triangles LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

find_program(GLSLC glslc
    HINTS "$ENV{VULKAN_SDK}/bin"
    REQUIRED
)

set(SHADER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

file(MAKE_DIRECTORY "${SHADER_BINARY_DIR}")

set(SHADER_SOURCES
    "${SHADER_SOURCE_DIR}/triangle.vert"
    "${SHADER_SOURCE_DIR}/triangle.frag"
)

set(SHADER_BINARIES)

foreach(SHADER IN LISTS SHADER_SOURCES)
    get_filename_component(SHADER_NAME "${SHADER}" NAME)
    set(SPIRV "${SHADER_BINARY_DIR}/${SHADER_NAME}.spv")

    add_custom_command(
        OUTPUT "${SPIRV}"
        COMMAND "${GLSLC}" "${SHADER}" -o "${SPIRV}"
        DEPENDS "${SHADER}"
        VERBATIM
    )

    list(APPEND SHADER_BINARIES "${SPIRV}")
endforeach()

add_custom_target(Shaders DEPENDS ${SHADER_BINARIES})

add_library(TriangleGeometry STATIC
    src/TriangleGenerator.cpp
)

target_include_directories(TriangleGeometry PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(TriangleGeometry PUBLIC cxx_std_17)

add_executable(Triangles
    src/main.cpp
    src/Renderer.cpp
)

add_dependencies(Triangles Shaders)

target_include_directories(Triangles PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_definitions(Triangles PRIVATE
    TRIANGLES_SHADER_DIR="${SHADER_BINARY_DIR}"
)

target_link_libraries(Triangles PRIVATE
    TriangleGeometry
    Vulkan::Vulkan
    glfw
)
