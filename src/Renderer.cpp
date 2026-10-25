#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "triangles/Renderer.hpp"
#include "triangles/Vertex.hpp"

#include <cstddef>

#include <algorithm>
#include <array>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef TRIANGLES_SHADER_DIR
#define TRIANGLES_SHADER_DIR "shaders"
#endif

namespace triangles {

namespace {

constexpr uint32_t WINDOW_WIDTH = 1280;
constexpr uint32_t WINDOW_HEIGHT = 720;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
constexpr float PI = 3.14159265358979323846f;

struct Mat4 { float m[16]{}; };
struct Vec3 { float x, y, z; };

Vec3 subtract(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
Vec3 normalize(Vec3 v) {
    const float len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    return {v.x/len, v.y/len, v.z/len};
}
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 out{};
    for (int col=0; col<4; ++col)
        for (int row=0; row<4; ++row)
            for (int k=0; k<4; ++k)
                out.m[col*4+row] += a.m[k*4+row] * b.m[col*4+k];
    return out;
}
Mat4 lookAt(Vec3 eye, Vec3 target, Vec3 up) {
    const Vec3 f = normalize(subtract(target, eye));
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);
    Mat4 m{};
    m.m[0]=s.x; m.m[4]=s.y; m.m[8]=s.z;
    m.m[1]=u.x; m.m[5]=u.y; m.m[9]=u.z;
    m.m[2]=-f.x; m.m[6]=-f.y; m.m[10]=-f.z;
    m.m[12]=-dot(s,eye); m.m[13]=-dot(u,eye); m.m[14]=dot(f,eye); m.m[15]=1.0f;
    return m;
}
Mat4 perspective(float fov, float aspect, float nearPlane, float farPlane) {
    Mat4 m{};
    const float f = 1.0f / std::tan(fov * 0.5f);
    m.m[0] = f / aspect;
    m.m[5] = -f; // Vulkan framebuffer Y points down.
    m.m[10] = farPlane / (nearPlane - farPlane);
    m.m[11] = -1.0f;
    m.m[14] = (farPlane * nearPlane) / (nearPlane - farPlane);
    return m;
}

const std::vector<const char*> DEVICE_EXTENSIONS = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct QueueFamilies {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;

    bool complete() const {
        return graphics.has_value() && present.has_value();
    }
};

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class VulkanRendererImpl {
public:
    void run(const std::vector<Vertex>& vertices) {
        vertices_ = vertices;
        vertexCount_ = static_cast<uint32_t>(vertices_.size());
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    std::vector<VkImage> depthImages;
    std::vector<VkDeviceMemory> depthImageMemories;
    std::vector<VkImageView> depthImageViews;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    std::vector<VkFramebuffer> framebuffers;
    VkFormat swapchainImageFormat{};
    VkExtent2D swapchainExtent{};

    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    std::array<VkBuffer, MAX_FRAMES_IN_FLIGHT> vertexBuffers{};
    std::array<VkDeviceMemory, MAX_FRAMES_IN_FLIGHT> vertexBufferMemories{};
    std::array<void*, MAX_FRAMES_IN_FLIGHT> mappedVertexBuffers{};
    std::array<uint32_t, MAX_FRAMES_IN_FLIGHT> frameVertexCounts{};
    std::vector<Vertex> vertices_;
    uint32_t vertexCount_ = 0;
    bool frustumCullingEnabled_ = false;
    bool benchmarkLogging_ = false;
    std::ofstream benchmarkFile_;
    std::chrono::steady_clock::time_point benchmarkStart_{};

    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> commandBuffers{};

    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> imageAvailable{};
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> renderFinished{};
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> inFlightFences{};

    std::vector<VkFence> imagesInFlight;
    size_t currentFrame = 0;
    bool framebufferResized = false;
    float cameraYaw = 0.0f;
    float cameraPitch = 0.20f;
    float cameraDistance = 12.0f;
    bool dragging = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;

    static void framebufferResizeCallback(GLFWwindow* window, int, int) {
        auto* renderer = static_cast<VulkanRendererImpl*>(glfwGetWindowUserPointer(window));
        renderer->framebufferResized = true;
    }
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int) {
        auto* r = static_cast<VulkanRendererImpl*>(glfwGetWindowUserPointer(window));
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            r->dragging = action == GLFW_PRESS;
            glfwGetCursorPos(window, &r->lastMouseX, &r->lastMouseY);
        }
    }
    static void cursorPositionCallback(GLFWwindow* window, double x, double y) {
        auto* r = static_cast<VulkanRendererImpl*>(glfwGetWindowUserPointer(window));
        if (!r->dragging) { r->lastMouseX=x; r->lastMouseY=y; return; }
        r->cameraYaw += static_cast<float>(x-r->lastMouseX) * 0.006f;
        r->cameraPitch += static_cast<float>(y-r->lastMouseY) * 0.006f;
        r->cameraPitch = std::clamp(r->cameraPitch, -1.45f, 1.45f);
        r->lastMouseX=x; r->lastMouseY=y;
    }
    static void scrollCallback(GLFWwindow* window, double, double yOffset) {
        auto* r = static_cast<VulkanRendererImpl*>(glfwGetWindowUserPointer(window));
        r->cameraDistance = std::clamp(r->cameraDistance * static_cast<float>(std::exp(-yOffset*0.12)), 2.0f, 40.0f);
    }
    static void keyCallback(GLFWwindow* window, int key, int, int action, int) {
        auto* r = static_cast<VulkanRendererImpl*>(glfwGetWindowUserPointer(window));
        if (key == GLFW_KEY_C && action == GLFW_PRESS) {
            r->frustumCullingEnabled_ = !r->frustumCullingEnabled_;
        } else if (key == GLFW_KEY_B && action == GLFW_PRESS) {
            if (!r->benchmarkLogging_) {
                r->benchmarkFile_.open("benchmark.csv", std::ios::out | std::ios::trunc);
                if (!r->benchmarkFile_) {
                    std::cerr << "Could not open benchmark.csv for writing.\n";
                    return;
                }
                r->benchmarkFile_ << "elapsed_seconds,triangles,visible_triangles,culling,fps,cpu_frame_ms\n";
                r->benchmarkStart_ = std::chrono::steady_clock::now();
                r->benchmarkLogging_ = true;
                std::cout << "Benchmark CSV recording started: benchmark.csv\n";
            } else {
                r->benchmarkLogging_ = false;
                r->benchmarkFile_.close();
                std::cout << "Benchmark CSV recording stopped.\n";
            }
        }
    }

    Mat4 cameraMvp() const {
        const float cp = std::cos(cameraPitch), sp = std::sin(cameraPitch);
        const Vec3 eye{cameraDistance * cp * std::sin(cameraYaw),
                       cameraDistance * sp,
                       cameraDistance * cp * std::cos(cameraYaw)};
        const Mat4 view = lookAt(eye, {0.0f,0.0f,0.0f}, {0.0f,1.0f,0.0f});
        const float aspect = static_cast<float>(swapchainExtent.width) /
                             static_cast<float>(std::max(swapchainExtent.height, 1u));
        const Mat4 projection = perspective(60.0f * PI / 180.0f, aspect, 0.1f, 100.0f);
        return multiply(projection, view);
    }

    struct ClipPosition { float x, y, z, w; };

    static ClipPosition transformToClip(const Mat4& m, const Vertex& v) {
        const float x = v.position[0], y = v.position[1], z = v.position[2];
        return {
            m.m[0]*x + m.m[4]*y + m.m[8]*z + m.m[12],
            m.m[1]*x + m.m[5]*y + m.m[9]*z + m.m[13],
            m.m[2]*x + m.m[6]*y + m.m[10]*z + m.m[14],
            m.m[3]*x + m.m[7]*y + m.m[11]*z + m.m[15]
        };
    }

    static bool triangleOutsideFrustum(const Mat4& m, const Vertex* tri) {
        ClipPosition p[3] = {
            transformToClip(m, tri[0]), transformToClip(m, tri[1]),
            transformToClip(m, tri[2])
        };
        // Vulkan clip volume: -w <= x,y <= w and 0 <= z <= w.
        for (int plane = 0; plane < 6; ++plane) {
            bool allOutside = true;
            for (const auto& v : p) {
                bool outside = false;
                switch (plane) {
                    case 0: outside = v.x < -v.w; break;
                    case 1: outside = v.x >  v.w; break;
                    case 2: outside = v.y < -v.w; break;
                    case 3: outside = v.y >  v.w; break;
                    case 4: outside = v.z < 0.0f; break;
                    case 5: outside = v.z > v.w; break;
                }
                if (!outside) { allOutside = false; break; }
            }
            if (allOutside) return true;
        }
        return false;
    }

    void updateFrameVertexBuffer(size_t frameIndex, const Mat4& mvp) {
        auto* destination = static_cast<Vertex*>(mappedVertexBuffers[frameIndex]);
        uint32_t count = 0;
        if (!frustumCullingEnabled_) {
            std::memcpy(destination, vertices_.data(), sizeof(Vertex) * vertices_.size());
            count = static_cast<uint32_t>(vertices_.size());
        } else {
            for (size_t i = 0; i + 2 < vertices_.size(); i += 3) {
                if (triangleOutsideFrustum(mvp, vertices_.data() + i)) continue;
                std::memcpy(destination + count, vertices_.data() + i, sizeof(Vertex) * 3);
                count += 3;
            }
        }
        frameVertexCounts[frameIndex] = count;
    }

    static void check(VkResult result, const char* operation) {
        if (result != VK_SUCCESS) {
            throw std::runtime_error(
                std::string(operation) + " failed (VkResult " +
                std::to_string(result) + ")");
        }
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error(
                "Cannot open shader: " + filename);
        }

        const auto fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(),
                  static_cast<std::streamsize>(fileSize));
        return buffer;
    }

    void initWindow() {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(
            WINDOW_WIDTH, WINDOW_HEIGHT,
            "Triangles - Vulkan", nullptr, nullptr);

        if (!window) {
            throw std::runtime_error("Failed to create GLFW window");
        }

        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPositionCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwSetKeyCallback(window, keyCallback);
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createVertexBuffer();
        createCommandPool();
        createSyncObjects();
        recreateSwapchain();
    }

    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Triangles";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions =
            glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        if (!glfwExtensions) {
            throw std::runtime_error(
                "GLFW could not provide Vulkan extensions");
        }

        std::vector<const char*> extensions(
            glfwExtensions,
            glfwExtensions + glfwExtensionCount);

        uint32_t availableCount = 0;
        check(vkEnumerateInstanceExtensionProperties(
                  nullptr, &availableCount, nullptr),
              "Enumerate instance extensions");

        std::vector<VkExtensionProperties> available(availableCount);
        check(vkEnumerateInstanceExtensionProperties(
                  nullptr, &availableCount, available.data()),
              "Enumerate instance extensions");

        const bool portabilityAvailable = std::any_of(
            available.begin(), available.end(),
            [](const auto& ext) {
                return std::strcmp(
                    ext.extensionName,
                    VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
            });

        VkInstanceCreateFlags flags = 0;

        if (portabilityAvailable) {
            extensions.push_back(
                VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.flags = flags;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount =
            static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        check(vkCreateInstance(&createInfo, nullptr, &instance),
              "Create Vulkan instance");
    }

    void createSurface() {
        check(glfwCreateWindowSurface(
                  instance, window, nullptr, &surface),
              "Create window surface");
    }

    QueueFamilies findQueueFamilies(VkPhysicalDevice candidate) {
        QueueFamilies families;

        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(
            candidate, &count, nullptr);

        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            candidate, &count, queues.data());

        for (uint32_t i = 0; i < count; ++i) {
            if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                families.graphics = i;
            }

            VkBool32 presentSupport = VK_FALSE;
            check(vkGetPhysicalDeviceSurfaceSupportKHR(
                      candidate, i, surface, &presentSupport),
                  "Check presentation support");

            if (presentSupport) {
                families.present = i;
            }

            if (families.complete()) {
                break;
            }
        }

        return families;
    }

    bool supportsDeviceExtensions(VkPhysicalDevice candidate) {
        uint32_t count = 0;
        check(vkEnumerateDeviceExtensionProperties(
                  candidate, nullptr, &count, nullptr),
              "Enumerate device extensions");

        std::vector<VkExtensionProperties> available(count);
        check(vkEnumerateDeviceExtensionProperties(
                  candidate, nullptr, &count, available.data()),
              "Enumerate device extensions");

        std::set<std::string> required(
            DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());

        for (const auto& ext : available) {
            required.erase(ext.extensionName);
        }

        // MoltenVK may expose portability subset as a device extension.
        // If exposed, it must also be enabled.
        return required.empty();
    }

    SwapchainSupport querySwapchainSupport(VkPhysicalDevice candidate) {
        SwapchainSupport support;

        check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                  candidate, surface, &support.capabilities),
              "Get surface capabilities");

        uint32_t count = 0;
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(
                  candidate, surface, &count, nullptr),
              "Get surface formats");

        if (count > 0) {
            support.formats.resize(count);
            check(vkGetPhysicalDeviceSurfaceFormatsKHR(
                      candidate, surface, &count,
                      support.formats.data()),
                  "Get surface formats");
        }

        count = 0;
        check(vkGetPhysicalDeviceSurfacePresentModesKHR(
                  candidate, surface, &count, nullptr),
              "Get present modes");

        if (count > 0) {
            support.presentModes.resize(count);
            check(vkGetPhysicalDeviceSurfacePresentModesKHR(
                      candidate, surface, &count,
                      support.presentModes.data()),
                  "Get present modes");
        }

        return support;
    }

    bool isDeviceSuitable(VkPhysicalDevice candidate) {
        const QueueFamilies families = findQueueFamilies(candidate);

        if (!families.complete() ||
            !supportsDeviceExtensions(candidate)) {
            return false;
        }

        const SwapchainSupport support =
            querySwapchainSupport(candidate);

        return !support.formats.empty() &&
               !support.presentModes.empty();
    }

    void pickPhysicalDevice() {
        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(
                  instance, &count, nullptr),
              "Enumerate physical devices");

        if (count == 0) {
            throw std::runtime_error(
                "No Vulkan-compatible GPU found");
        }

        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(
                  instance, &count, devices.data()),
              "Enumerate physical devices");

        for (VkPhysicalDevice candidate : devices) {
            if (isDeviceSuitable(candidate)) {
                physicalDevice = candidate;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error(
                "No suitable GPU with graphics and presentation support");
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        std::cout << "Using GPU: " << properties.deviceName << '\n';
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1u << i)) &&
                (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable Vulkan memory type");
    }

    void createVertexBuffer() {
        if (vertices_.empty()) throw std::runtime_error("Cannot create an empty vertex buffer");
        const VkDeviceSize bufferSize = sizeof(Vertex) * vertices_.size();
        for (int frame = 0; frame < MAX_FRAMES_IN_FLIGHT; ++frame) {
            VkBufferCreateInfo bufferInfo{};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = bufferSize;
            bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            check(vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffers[frame]), "Create vertex buffer");
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device, vertexBuffers[frame], &requirements);
            VkMemoryAllocateInfo allocation{};
            allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            check(vkAllocateMemory(device, &allocation, nullptr, &vertexBufferMemories[frame]), "Allocate vertex buffer memory");
            check(vkBindBufferMemory(device, vertexBuffers[frame], vertexBufferMemories[frame], 0), "Bind vertex buffer memory");
            check(vkMapMemory(device, vertexBufferMemories[frame], 0, bufferSize, 0,
                              &mappedVertexBuffers[frame]), "Map vertex buffer memory");
        }
    }

    void createLogicalDevice() {
        const QueueFamilies families =
            findQueueFamilies(physicalDevice);

        std::set<uint32_t> uniqueFamilies = {
            families.graphics.value(),
            families.present.value()
        };

        const float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueInfos;

        for (uint32_t family : uniqueFamilies) {
            VkDeviceQueueCreateInfo queueInfo{};
            queueInfo.sType =
                VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInfo.queueFamilyIndex = family;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;
            queueInfos.push_back(queueInfo);
        }

        uint32_t extensionCount = 0;
        check(vkEnumerateDeviceExtensionProperties(
                  physicalDevice, nullptr, &extensionCount, nullptr),
              "Enumerate device extensions");

        std::vector<VkExtensionProperties> available(extensionCount);
        check(vkEnumerateDeviceExtensionProperties(
                  physicalDevice, nullptr, &extensionCount,
                  available.data()),
              "Enumerate device extensions");

        std::vector<const char*> extensions = DEVICE_EXTENSIONS;

        for (const auto& ext : available) {
            if (std::strcmp(
                    ext.extensionName,
                    "VK_KHR_portability_subset") == 0) {
                extensions.push_back("VK_KHR_portability_subset");
                break;
            }
        }

        VkPhysicalDeviceFeatures features{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount =
            static_cast<uint32_t>(queueInfos.size());
        createInfo.pQueueCreateInfos = queueInfos.data();
        createInfo.pEnabledFeatures = &features;
        createInfo.enabledExtensionCount =
            static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        check(vkCreateDevice(
                  physicalDevice, &createInfo, nullptr, &device),
              "Create logical device");

        vkGetDeviceQueue(device, families.graphics.value(),
                         0, &graphicsQueue);
        vkGetDeviceQueue(device, families.present.value(),
                         0, &presentQueue);
    }

    VkSurfaceFormatKHR chooseSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR>& formats) {
        for (const auto& format : formats) {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return format;
            }
        }

        return formats.front();
    }

    VkPresentModeKHR choosePresentMode(
        const std::vector<VkPresentModeKHR>& modes) {
        // FIFO is guaranteed to be supported by Vulkan.
        for (const auto mode : modes) {
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return mode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseExtent(
        const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width !=
            std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D extent{
            static_cast<uint32_t>(std::max(width, 0)),
            static_cast<uint32_t>(std::max(height, 0))
        };

        extent.width = std::clamp(
            extent.width,
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);

        extent.height = std::clamp(
            extent.height,
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);

        return extent;
    }

    void createSwapchain() {
        const SwapchainSupport support =
            querySwapchainSupport(physicalDevice);

        const VkSurfaceFormatKHR surfaceFormat =
            chooseSurfaceFormat(support.formats);
        const VkPresentModeKHR presentMode =
            choosePresentMode(support.presentModes);
        const VkExtent2D extent =
            chooseExtent(support.capabilities);

        uint32_t imageCount =
            support.capabilities.minImageCount + 1;

        if (support.capabilities.maxImageCount > 0) {
            imageCount = std::min(
                imageCount,
                support.capabilities.maxImageCount);
        }

        const QueueFamilies families =
            findQueueFamilies(physicalDevice);

        const uint32_t queueIndices[] = {
            families.graphics.value(),
            families.present.value()
        };

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType =
            VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        if (families.graphics != families.present) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        check(vkCreateSwapchainKHR(
                  device, &createInfo, nullptr, &swapchain),
              "Create swapchain");

        check(vkGetSwapchainImagesKHR(
                  device, swapchain, &imageCount, nullptr),
              "Get swapchain images");

        swapchainImages.resize(imageCount);
        check(vkGetSwapchainImagesKHR(
                  device, swapchain, &imageCount,
                  swapchainImages.data()),
              "Get swapchain images");

        swapchainImageFormat = surfaceFormat.format;
        swapchainExtent = extent;
    }

    void createImageViews() {
        swapchainImageViews.resize(swapchainImages.size());

        for (size_t i = 0; i < swapchainImages.size(); ++i) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType =
                VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapchainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapchainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask =
                VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            check(vkCreateImageView(
                      device, &createInfo, nullptr,
                      &swapchainImageViews[i]),
                  "Create image view");
        }
    }

    VkShaderModule createShaderModule(
        const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode =
            reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule module = VK_NULL_HANDLE;
        check(vkCreateShaderModule(
                  device, &createInfo, nullptr, &module),
              "Create shader module");
        return module;
    }

    VkFormat findDepthFormat() const {
        const std::array<VkFormat, 3> candidates = {
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT
        };

        for (VkFormat format : candidates) {
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
            if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                return format;
            }
        }
        throw std::runtime_error("No supported depth attachment format found");
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspect;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView view = VK_NULL_HANDLE;
        check(vkCreateImageView(device, &viewInfo, nullptr, &view), "Create image view");
        return view;
    }

    void createDepthResources() {
        depthFormat = findDepthFormat();
        const size_t count = swapchainImages.size();
        depthImages.resize(count, VK_NULL_HANDLE);
        depthImageMemories.resize(count, VK_NULL_HANDLE);
        depthImageViews.resize(count, VK_NULL_HANDLE);

        for (size_t i = 0; i < count; ++i) {
            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.extent = {swapchainExtent.width, swapchainExtent.height, 1};
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = depthFormat;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            check(vkCreateImage(device, &imageInfo, nullptr, &depthImages[i]), "Create depth image");

            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(device, depthImages[i], &requirements);
            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = requirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(
                requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            check(vkAllocateMemory(device, &allocInfo, nullptr, &depthImageMemories[i]), "Allocate depth image memory");
            check(vkBindImageMemory(device, depthImages[i], depthImageMemories[i], 0), "Bind depth image memory");
            depthImageViews[i] = createImageView(
                depthImages[i], depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
        }
    }

    void createRenderPass() {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapchainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout =
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = depthFormat;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorReference{};
        colorReference.attachment = 0;
        colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentReference depthReference{};
        depthReference.attachment = 1;
        depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorReference;
        subpass.pDepthStencilAttachment = &depthReference;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        const std::array<VkAttachmentDescription, 2> attachments = {
            colorAttachment, depthAttachment
        };
        VkRenderPassCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        createInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        createInfo.pAttachments = attachments.data();
        createInfo.subpassCount = 1;
        createInfo.pSubpasses = &subpass;
        createInfo.dependencyCount = 1;
        createInfo.pDependencies = &dependency;

        check(vkCreateRenderPass(
                  device, &createInfo, nullptr, &renderPass),
              "Create render pass");
    }

    void createGraphicsPipeline() {
        const auto vertCode = readFile(
            std::string(TRIANGLES_SHADER_DIR) + "/triangle.vert.spv");
        const auto fragCode = readFile(
            std::string(TRIANGLES_SHADER_DIR) + "/triangle.frag.spv");

        VkShaderModule vertModule = createShaderModule(vertCode);
        VkShaderModule fragModule = createShaderModule(fragCode);

        VkPipelineShaderStageCreateInfo vertStage{};
        vertStage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertStage.module = vertModule;
        vertStage.pName = "main";

        VkPipelineShaderStageCreateInfo fragStage{};
        fragStage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStage.module = fragModule;
        fragStage.pName = "main";

        VkPipelineShaderStageCreateInfo stages[] = {
            vertStage, fragStage
        };

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 2> attributes{};
        attributes[0].binding = 0;
        attributes[0].location = 0;
        attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[0].offset = offsetof(Vertex, position);
        attributes[1].binding = 0;
        attributes[1].location = 1;
        attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributes[1].offset = offsetof(Vertex, color);

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount =
            static_cast<uint32_t>(attributes.size());
        vertexInput.pVertexAttributeDescriptions = attributes.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType =
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType =
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        const VkDynamicState dynamicStates[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType =
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(Mat4);

        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushRange;

        check(vkCreatePipelineLayout(
                  device, &layoutInfo, nullptr, &pipelineLayout),
              "Create pipeline layout");

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType =
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = stages;
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        const VkResult result = vkCreateGraphicsPipelines(
            device, VK_NULL_HANDLE, 1, &pipelineInfo,
            nullptr, &graphicsPipeline);

        vkDestroyShaderModule(device, fragModule, nullptr);
        vkDestroyShaderModule(device, vertModule, nullptr);

        check(result, "Create graphics pipeline");
    }

    void createFramebuffers() {
        framebuffers.resize(swapchainImageViews.size());

        for (size_t i = 0; i < swapchainImageViews.size(); ++i) {
            VkImageView attachments[] = {
                swapchainImageViews[i],
                depthImageViews[i]
            };

            VkFramebufferCreateInfo createInfo{};
            createInfo.sType =
                VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            createInfo.renderPass = renderPass;
            createInfo.attachmentCount = 2;
            createInfo.pAttachments = attachments;
            createInfo.width = swapchainExtent.width;
            createInfo.height = swapchainExtent.height;
            createInfo.layers = 1;

            check(vkCreateFramebuffer(
                      device, &createInfo, nullptr, &framebuffers[i]),
                  "Create framebuffer");
        }
    }

    void createCommandPool() {
        const QueueFamilies families =
            findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo createInfo{};
        createInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        createInfo.flags =
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        createInfo.queueFamilyIndex = families.graphics.value();

        check(vkCreateCommandPool(
                  device, &createInfo, nullptr, &commandPool),
              "Create command pool");

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

        check(vkAllocateCommandBuffers(
                  device, &allocInfo, commandBuffers.data()),
              "Allocate command buffers");
    }

    void createSyncObjects() {
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType =
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            check(vkCreateSemaphore(
                      device, &semaphoreInfo, nullptr,
                      &imageAvailable[i]),
                  "Create image-available semaphore");

            check(vkCreateSemaphore(
                      device, &semaphoreInfo, nullptr,
                      &renderFinished[i]),
                  "Create render-finished semaphore");

            check(vkCreateFence(
                      device, &fenceInfo, nullptr,
                      &inFlightFences[i]),
                  "Create in-flight fence");
        }
    }

    void cleanupSwapchain() {
        for (VkFramebuffer framebuffer : framebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        framebuffers.clear();

        for (VkImageView view : depthImageViews) {
            if (view != VK_NULL_HANDLE) vkDestroyImageView(device, view, nullptr);
        }
        depthImageViews.clear();
        for (VkImage image : depthImages) {
            if (image != VK_NULL_HANDLE) vkDestroyImage(device, image, nullptr);
        }
        depthImages.clear();
        for (VkDeviceMemory memory : depthImageMemories) {
            if (memory != VK_NULL_HANDLE) vkFreeMemory(device, memory, nullptr);
        }
        depthImageMemories.clear();

        if (graphicsPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, graphicsPipeline, nullptr);
            graphicsPipeline = VK_NULL_HANDLE;
        }

        if (pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            pipelineLayout = VK_NULL_HANDLE;
        }

        if (renderPass != VK_NULL_HANDLE) {
            vkDestroyRenderPass(device, renderPass, nullptr);
            renderPass = VK_NULL_HANDLE;
        }

        for (VkImageView imageView : swapchainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }
        swapchainImageViews.clear();
        swapchainImages.clear();

        if (swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device, swapchain, nullptr);
            swapchain = VK_NULL_HANDLE;
        }

        imagesInFlight.clear();
    }

    void recreateSwapchain() {
        int width = 0;
        int height = 0;

        glfwGetFramebufferSize(window, &width, &height);

        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();

            if (glfwWindowShouldClose(window)) {
                return;
            }
        }

        check(vkDeviceWaitIdle(device), "Wait for device idle");

        cleanupSwapchain();
        createSwapchain();
        createImageViews();
        createDepthResources();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();

        imagesInFlight.assign(
            swapchainImages.size(), VK_NULL_HANDLE);

        framebufferResized = false;
    }

    void recordCommandBuffer(
        VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        check(vkBeginCommandBuffer(commandBuffer, &beginInfo),
              "Begin command buffer");

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.025f, 0.035f, 0.06f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo renderInfo{};
        renderInfo.sType =
            VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderInfo.renderPass = renderPass;
        renderInfo.framebuffer = framebuffers[imageIndex];
        renderInfo.renderArea.offset = {0, 0};
        renderInfo.renderArea.extent = swapchainExtent;
        renderInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(
            commandBuffer, &renderInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapchainExtent.width);
        viewport.height = static_cast<float>(swapchainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapchainExtent;

        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffers[currentFrame], &offset);
        const Mat4 mvp = cameraMvp();
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(Mat4), &mvp);
        vkCmdDraw(commandBuffer, frameVertexCounts[currentFrame], 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        check(vkEndCommandBuffer(commandBuffer),
              "End command buffer");
    }

    void drawFrame() {
        check(vkWaitForFences(
                  device, 1, &inFlightFences[currentFrame],
                  VK_TRUE, UINT64_MAX),
              "Wait for frame fence");

        const Mat4 frameMvp = cameraMvp();
        updateFrameVertexBuffer(currentFrame, frameMvp);

        uint32_t imageIndex = 0;
        const VkResult acquireResult = vkAcquireNextImageKHR(
            device, swapchain, UINT64_MAX,
            imageAvailable[currentFrame], VK_NULL_HANDLE,
            &imageIndex);

        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }

        if (acquireResult != VK_SUCCESS &&
            acquireResult != VK_SUBOPTIMAL_KHR) {
            check(acquireResult, "Acquire swapchain image");
        }

        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            check(vkWaitForFences(
                      device, 1, &imagesInFlight[imageIndex],
                      VK_TRUE, UINT64_MAX),
                  "Wait for swapchain image fence");
        }

        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        check(vkResetFences(
                  device, 1, &inFlightFences[currentFrame]),
              "Reset frame fence");

        check(vkResetCommandBuffer(
                  commandBuffers[currentFrame], 0),
              "Reset command buffer");

        recordCommandBuffer(
            commandBuffers[currentFrame], imageIndex);

        VkPipelineStageFlags waitStage =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailable[currentFrame];
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderFinished[currentFrame];

        check(vkQueueSubmit(
                  graphicsQueue, 1, &submitInfo,
                  inFlightFences[currentFrame]),
              "Submit draw command");

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinished[currentFrame];
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIndex;

        const VkResult presentResult =
            vkQueuePresentKHR(presentQueue, &presentInfo);

        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
            presentResult == VK_SUBOPTIMAL_KHR ||
            framebufferResized) {
            recreateSwapchain();
        } else {
            check(presentResult, "Present swapchain image");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void mainLoop() {
        using Clock = std::chrono::steady_clock;
        auto previousFrame = Clock::now();
        auto statsStart = previousFrame;
        double accumulatedFrameMs = 0.0;
        uint64_t framesInInterval = 0;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();

            const auto now = Clock::now();
            const double frameMs =
                std::chrono::duration<double, std::milli>(now - previousFrame).count();
            previousFrame = now;
            accumulatedFrameMs += frameMs;
            ++framesInInterval;

            const double intervalSeconds =
                std::chrono::duration<double>(now - statsStart).count();
            if (intervalSeconds >= 0.5 && framesInInterval > 0) {
                const double fps = static_cast<double>(framesInInterval) / intervalSeconds;
                const double averageFrameMs =
                    accumulatedFrameMs / static_cast<double>(framesInInterval);

                std::ostringstream title;
                title << "Triangles | FPS: " << std::fixed << std::setprecision(1)
                      << fps << " | CPU frame: " << std::setprecision(2)
                      << averageFrameMs << " ms | Triangles: " << (vertexCount_ / 3)
                      << " | Visible: " << (frameVertexCounts[currentFrame] / 3)
                      << " | Culling: " << (frustumCullingEnabled_ ? "ON" : "OFF")
                      << " (C)";
                glfwSetWindowTitle(window, title.str().c_str());

                if (benchmarkLogging_ && benchmarkFile_) {
                    const double elapsed = std::chrono::duration<double>(now - benchmarkStart_).count();
                    benchmarkFile_ << std::fixed << std::setprecision(3)
                                   << elapsed << ',' << (vertexCount_ / 3) << ','
                                   << (frameVertexCounts[currentFrame] / 3) << ','
                                   << (frustumCullingEnabled_ ? 1 : 0) << ','
                                   << std::setprecision(2) << fps << ',' << averageFrameMs << '\n';
                    benchmarkFile_.flush();
                }

                statsStart = now;
                accumulatedFrameMs = 0.0;
                framesInInterval = 0;
            }
        }

        check(vkDeviceWaitIdle(device), "Wait before shutdown");
    }

    void cleanup() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            cleanupSwapchain();

            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
                if (imageAvailable[i] != VK_NULL_HANDLE) {
                    vkDestroySemaphore(
                        device, imageAvailable[i], nullptr);
                }
                if (renderFinished[i] != VK_NULL_HANDLE) {
                    vkDestroySemaphore(
                        device, renderFinished[i], nullptr);
                }
                if (inFlightFences[i] != VK_NULL_HANDLE) {
                    vkDestroyFence(
                        device, inFlightFences[i], nullptr);
                }
            }

            if (commandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, commandPool, nullptr);
            }
            for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
                if (mappedVertexBuffers[i] != nullptr && vertexBufferMemories[i] != VK_NULL_HANDLE)
                    vkUnmapMemory(device, vertexBufferMemories[i]);
                if (vertexBuffers[i] != VK_NULL_HANDLE)
                    vkDestroyBuffer(device, vertexBuffers[i], nullptr);
                if (vertexBufferMemories[i] != VK_NULL_HANDLE)
                    vkFreeMemory(device, vertexBufferMemories[i], nullptr);
            }

            vkDestroyDevice(device, nullptr);
        }

        if (surface != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance, surface, nullptr);
        }

        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }

        if (window) {
            glfwDestroyWindow(window);
        }

        glfwTerminate();
    }
};

} // namespace

class Renderer::Impl {
public:
    void run(const std::vector<Vertex>& vertices) {
        implementation_.run(vertices);
    }

private:
    VulkanRendererImpl implementation_;
};

Renderer::Renderer() : impl_(std::make_unique<Impl>()) {}
Renderer::~Renderer() = default;

void Renderer::run(const std::vector<Vertex>& vertices) {
    impl_->run(vertices);
}

} // namespace triangles
