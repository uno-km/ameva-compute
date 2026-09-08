#include "ac_vk_loader.h"
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <windows.h>
#define DLOPEN(path) (void*)LoadLibraryA(path)
#define DLSYM(handle, name) (void*)GetProcAddress((HMODULE)handle, name)
#define DLCLOSE(handle) FreeLibrary((HMODULE)handle)
#else
#include <dlfcn.h>
#define DLOPEN(path) dlopen(path, RTLD_LAZY | RTLD_LOCAL)
#define DLSYM(handle, name) dlsym(handle, name)
#define DLCLOSE(handle) dlclose(handle)
#endif

namespace ameva {
namespace backend {
namespace vk {

VulkanLoader& VulkanLoader::Instance() {
    static VulkanLoader instance;
    return instance;
}

VulkanLoader::~VulkanLoader() {
    if (handle_) {
        DLCLOSE(handle_);
        handle_ = nullptr;
    }
}

bool VulkanLoader::EnsureLoaded(const std::string& explicit_path) {
    if (handle_ != nullptr) return true;

    std::vector<std::string> candidates;
    if (!explicit_path.empty()) {
        candidates.push_back(explicit_path);
    }
#if defined(__ANDROID__)
    candidates.push_back("/system/lib64/libvulkan.so");
    candidates.push_back("/vendor/lib64/hw/vulkan.adreno.so");
    candidates.push_back("/vendor/lib64/libvulkan.so");
    candidates.push_back("libvulkan.so");
#elif defined(_WIN32)
    candidates.push_back("vulkan-1.dll");
#else
    candidates.push_back("libvulkan.so.1");
    candidates.push_back("libvulkan.so");
#endif

    for (const auto& path : candidates) {
        handle_ = DLOPEN(path.c_str());
        if (handle_) {
            loaded_path_ = path;
            return true;
        }
    }
    return false;
}

void* VulkanLoader::GetProcAddress(const char* name) const {
    if (!handle_ || !name) return nullptr;
    return DLSYM(handle_, name);
}

bool VulkanLoader::IsAvailable() const {
    return handle_ != nullptr;
}

const std::string& VulkanLoader::GetLoadedPath() const {
    return loaded_path_;
}

} // namespace vk
} // namespace backend
} // namespace ameva
