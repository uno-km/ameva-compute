// Standalone CLI harness for ameva_llm_api.h vision path (Qwen2-VL + mmproj).
// Not part of the shipped app; built only to compare GPU (ggml-ameva) vs CPU backend output
// on-device without an APK. No silent CPU fallback: ameva_llm_load() is called once per backend
// argument and any failure is a hard, visible error (AMEVA_LLM_ERR_*), not a retried/caught one.
#include "ameva_llm_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static const char* statusName(AmevaLlmStatus s) {
    switch (s) {
        case AMEVA_LLM_OK: return "AMEVA_LLM_OK";
        case AMEVA_LLM_ERR_BACKEND_LOAD_FAILED: return "AMEVA_LLM_ERR_BACKEND_LOAD_FAILED";
        case AMEVA_LLM_ERR_BACKEND_NO_DEVICE: return "AMEVA_LLM_ERR_BACKEND_NO_DEVICE";
        case AMEVA_LLM_ERR_MODEL_LOAD_FAILED: return "AMEVA_LLM_ERR_MODEL_LOAD_FAILED";
        case AMEVA_LLM_ERR_CONTEXT_CREATE_FAILED: return "AMEVA_LLM_ERR_CONTEXT_CREATE_FAILED";
        case AMEVA_LLM_ERR_TOKENIZE_FAILED: return "AMEVA_LLM_ERR_TOKENIZE_FAILED";
        case AMEVA_LLM_ERR_DECODE_FAILED: return "AMEVA_LLM_ERR_DECODE_FAILED";
        case AMEVA_LLM_ERR_OUTPUT_BUFFER_TOO_SMALL: return "AMEVA_LLM_ERR_OUTPUT_BUFFER_TOO_SMALL";
        case AMEVA_LLM_ERR_INVALID_ARGUMENT: return "AMEVA_LLM_ERR_INVALID_ARGUMENT";
        case AMEVA_LLM_ERR_VISION_LOAD_FAILED: return "AMEVA_LLM_ERR_VISION_LOAD_FAILED";
        case AMEVA_LLM_ERR_VISION_NOT_LOADED: return "AMEVA_LLM_ERR_VISION_NOT_LOADED";
        case AMEVA_LLM_ERR_IMAGE_LOAD_FAILED: return "AMEVA_LLM_ERR_IMAGE_LOAD_FAILED";
        case AMEVA_LLM_ERR_VISION_TOKENIZE_FAILED: return "AMEVA_LLM_ERR_VISION_TOKENIZE_FAILED";
        case AMEVA_LLM_ERR_VISION_ENCODE_FAILED: return "AMEVA_LLM_ERR_VISION_ENCODE_FAILED";
        case AMEVA_LLM_ERR_GPU_ONLY_UNSUPPORTED_OP: return "AMEVA_LLM_ERR_GPU_ONLY_UNSUPPORTED_OP";
        case AMEVA_LLM_ERR_GPU_BUDGET_EXCEEDED: return "AMEVA_LLM_ERR_GPU_BUDGET_EXCEEDED";
        default: return "UNKNOWN";
    }
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <mmproj.gguf> <image.jpg> <backend: cpu|vulkan|opencl> "
            "<ggml-ameva.so path or \"-\"> [budget_mode: default|gpu_only|gpu_then_cpu|cpu_only] "
            "[budget_fraction]\n",
            argv[0]);
        return 2;
    }
    const char* model_path = argv[1];
    const char* mmproj_path = argv[2];
    const char* image_path = argv[3];
    const char* backend = argv[4];
    const char* backend_plugin_path = (std::strcmp(argv[5], "-") == 0) ? nullptr : argv[5];
    const char* budget_mode_arg = (argc > 6) ? argv[6] : "default";
    const float budget_fraction = (argc > 7) ? std::strtof(argv[7], nullptr) : 0.0f;

    AmevaGpuBudgetOptions budget_opts = {};
    budget_opts.budget_fraction = budget_fraction;
    if (std::strcmp(budget_mode_arg, "gpu_only") == 0) budget_opts.mode = AMEVA_GPU_BUDGET_MODE_GPU_ONLY;
    else if (std::strcmp(budget_mode_arg, "gpu_then_cpu") == 0) budget_opts.mode = AMEVA_GPU_BUDGET_MODE_GPU_THEN_CPU;
    else if (std::strcmp(budget_mode_arg, "cpu_only") == 0) budget_opts.mode = AMEVA_GPU_BUDGET_MODE_CPU_ONLY;
    else budget_opts.mode = AMEVA_GPU_BUDGET_MODE_DEFAULT;
    const AmevaGpuBudgetOptions* budget_opts_ptr =
        (budget_opts.mode == AMEVA_GPU_BUDGET_MODE_DEFAULT) ? nullptr : &budget_opts;

    char err[512] = {0};
    AmevaLlm* llm = nullptr;

    std::printf("[ameva_vision_test] backend=%s model=%s budget_mode=%s\n", backend, model_path, budget_mode_arg);
    AmevaLlmStatus st = ameva_llm_load(model_path, backend_plugin_path, /*cache_dir=*/nullptr,
                                       backend, /*n_ctx=*/4096, budget_opts_ptr, &llm, err, sizeof(err));
    if (st != AMEVA_LLM_OK) {
        std::printf("RESULT: FAIL\nSTAGE: ameva_llm_load\nSTATUS: %s\nERR: %s\n", statusName(st), err);
        return 1;
    }
    std::printf("[ameva_vision_test] model loaded OK\n");

    st = ameva_llm_load_mmproj(llm, mmproj_path, err, sizeof(err));
    if (st != AMEVA_LLM_OK) {
        std::printf("RESULT: FAIL\nSTAGE: ameva_llm_load_mmproj\nSTATUS: %s\nERR: %s\n", statusName(st), err);
        ameva_llm_free(llm);
        return 1;
    }
    std::printf("[ameva_vision_test] mmproj loaded OK, supports_vision=%d\n",
                (int)ameva_llm_supports_vision(llm));

    char out[8192] = {0};
    st = ameva_llm_generate_with_image(llm, image_path, "Describe this image in one sentence.",
                                       /*max_tokens=*/128, /*temperature=*/0.0f, /*seed=*/42,
                                       out, sizeof(out), err, sizeof(err));
    if (st != AMEVA_LLM_OK) {
        std::printf("RESULT: FAIL\nSTAGE: ameva_llm_generate_with_image\nSTATUS: %s\nERR: %s\n", statusName(st), err);
        ameva_llm_free(llm);
        return 1;
    }

    std::printf("RESULT: PASS\nOUTPUT: %s\n", out);
    ameva_llm_free(llm);
    return 0;
}
