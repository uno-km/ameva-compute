/**
 * @file test_formal_backend_smoke.cpp
 * @brief Smoke verification of formal AMEVA ggml_backend interface
 */

#include "ggml-backend-ameva.h"
#include "../../third_party/whisper.cpp/ggml/src/ggml-backend-impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main() {
    printf("=== [AMEVA FORMAL GGML BACKEND SMOKE TEST] ===\n");

    // 1. Backend registration
    ameva_backend_register();
    ggml_backend_reg_t reg = ggml_backend_ameva_reg();
    if (!reg) {
        fprintf(stderr, "FATAL: ggml_backend_ameva_reg returned NULL\n");
        return 1;
    }
    printf("AMEVA_BACKEND_REGISTERED=true\n");

    // 2. Device count and handle
    size_t dev_count = reg->iface.get_device_count(reg);
    printf("AMEVA_BACKEND_DEVICE_COUNT=%zu\n", dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "FATAL: Expected at least 1 AMEVA backend device\n");
        return 2;
    }

    ggml_backend_dev_t dev = reg->iface.get_device(reg, 0);
    if (!dev) {
        fprintf(stderr, "FATAL: Failed to get AMEVA device 0\n");
        return 3;
    }
    printf("AMEVA_DEVICE_NAME=%s\n", dev->iface.get_name(dev));
    printf("AMEVA_DEVICE_DESC=%s\n", dev->iface.get_description(dev));
    printf("AMEVA_DEVICE_TYPE=%d (ACCEL=3)\n", (int)dev->iface.get_type(dev));

    // 3. Buffer allocation smoke
    ggml_backend_buffer_type_t buft = dev->iface.get_buffer_type(dev);
    if (!buft) {
        fprintf(stderr, "FATAL: Failed to get AMEVA buffer type\n");
        return 4;
    }
    printf("AMEVA_BUFT_NAME=%s\n", buft->iface.get_name(buft));

    const size_t test_size = 1024 * 1024; // 1MB
    ggml_backend_buffer_t buf = buft->iface.alloc_buffer(buft, test_size);
    if (!buf) {
        fprintf(stderr, "FATAL: Failed to allocate AMEVA buffer\n");
        return 5;
    }
    printf("AMEVA_BUFFER_ALLOC_SIZE=%zu\n", test_size);
    printf("AMEVA_BUFFER_BASE=%p\n", buf->iface.get_base(buf));
    printf("AMEVA_BUFFER_SMOKE=PASS\n");

    // 4. Backend (stream) initialization
    ggml_backend_t backend = dev->iface.init_backend(dev, nullptr);
    if (!backend) {
        fprintf(stderr, "FATAL: Failed to initialize AMEVA backend stream\n");
        buf->iface.free_buffer(buf);
        return 6;
    }
    printf("AMEVA_BACKEND_INIT=PASS\n");

    // 5. Operator support query
    struct ggml_init_params params = { 16 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    // Target op: M=1500, N=384, K=384 (Whisper Tiny encoder projection)
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 384, 384);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 384, 1500);
    struct ggml_tensor * target_op = ggml_mul_mat(ctx, w, a);

    bool supported = dev->iface.supports_op(dev, target_op);
    printf("AMEVA_OP_MUL_MAT_SUPPORTED=%s\n", supported ? "true" : "false");
    if (!supported) {
        fprintf(stderr, "FATAL: Expected target MatMul to be supported\n");
        ggml_free(ctx);
        backend->iface.free(backend);
        buf->iface.free_buffer(buf);
        return 7;
    }

    // Unsupported op query: ADD (must return false)
    struct ggml_tensor * dummy_add = ggml_add(ctx, a, a);
    bool add_supported = dev->iface.supports_op(dev, dummy_add);
    printf("AMEVA_OP_ADD_SUPPORTED=%s\n", add_supported ? "true" : "false");
    if (add_supported) {
        fprintf(stderr, "FATAL: Unsupported OP returned true! Violates fail-fast contract.\n");
        ggml_free(ctx);
        backend->iface.free(backend);
        buf->iface.free_buffer(buf);
        return 8;
    }

    // 6. Cleanup
    ggml_free(ctx);
    backend->iface.free(backend);
    buf->iface.free_buffer(buf);

    printf("=== [AMEVA FORMAL BACKEND SMOKE PASSED] ===\n");
    return 0;
}
