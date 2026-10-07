/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#pragma once
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace ameva {
namespace backend {
namespace vk {

class VulkanLoader {
public:
    static VulkanLoader& Instance();
    bool EnsureLoaded(const std::string& explicit_path = "");
    void* GetProcAddress(const char* name) const;
    bool IsAvailable() const;
    const std::string& GetLoadedPath() const;

private:
    VulkanLoader() = default;
    ~VulkanLoader();
    VulkanLoader(const VulkanLoader&) = delete;
    VulkanLoader& operator=(const VulkanLoader&) = delete;

    void* handle_{nullptr};
    std::string loaded_path_;
};

} // namespace vk
} // namespace backend
} // namespace ameva
