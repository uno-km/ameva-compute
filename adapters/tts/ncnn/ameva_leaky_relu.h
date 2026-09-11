#ifndef AMEVA_LEAKY_RELU_H
#define AMEVA_LEAKY_RELU_H

#include <layer.h>

#if NCNN_VULKAN
#include <gpu.h>
#endif

namespace ameva {
namespace tts {

class AmevaLeakyReLU : public ncnn::Layer {
public:
    AmevaLeakyReLU();
    virtual ~AmevaLeakyReLU();

    virtual int load_param(const ncnn::ParamDict& pd) override;

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const override;

#if NCNN_VULKAN
    virtual int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const override;
#endif

private:
    float negative_slope_{0.1f};
};

ncnn::Layer* AmevaLeakyReLU_layer_creator(void* userdata);
void AmevaLeakyReLU_layer_destroyer(ncnn::Layer* layer, void* userdata);

} // namespace tts
} // namespace ameva

#endif // AMEVA_LEAKY_RELU_H
