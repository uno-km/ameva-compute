/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
#pragma once
// Internal to libggml-ameva: declarations shared by the buffer implementation (ggml-ameva-buffer.cpp), the graph
// engine (ggml-ameva.cpp) and the device (ggml-ameva-device.cpp). Not installed, not part of the plugin ABI.
#include "ggml.h"
#include "ac_runtime.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

// A tensor stored in the AMEVA repacked weight layout (ggml-ameva-repack.h), by its bytes inside the buffer.
struct AmevaRepackSpan {
    uint64_t off;    // tensor start inside the buffer
    uint64_t bytes;  // ggml_nbytes
    enum ggml_type type;
    int64_t K;       // weights per row
    int64_t rows;
};

// buffer->context of every AMEVA buffer
struct ggml_backend_ameva_buffer_context {
    AcBuffer* ac_buf;
    // ac_runtime_buffer_get_address: the base of tensor->data inside this buffer, for offsets only. The host reaches
    // the bytes through ac_runtime_buffer_write / read / fill (on OpenCL the address is an SVM pointer the host must
    // not dereference).
    void* base;
    size_t size;
    std::mutex mu;                          // guards repacked
    std::vector<AmevaRepackSpan> repacked;  // every repacked tensor of the buffer, in the order it was repacked
};

// ---- repacked weights (ggml-ameva-buffer.cpp) ----
// Weights are repacked unless AMEVA_REPACK=0. With repacking, AMEVA buffers are not host buffers: only the buffer
// interface (which converts) may read or write their tensors.
bool ameva_repack_enabled(void);
// AMEVA_BACKEND=opencl: the context is AC_BACKEND_OPENCL (Vulkan otherwise); its buffers are never host buffers
bool ameva_backend_is_opencl(void);
// AMEVA_REPACK_ALL=1 (tests): eligible quantized tensors of every buffer are repacked at their first write
bool ameva_repack_all_enabled(void);
// The repacked tensor holding t's bytes: t itself, or the tensor t is a view of. nullptr: t's bytes are in the ggml
// layout.
const struct ggml_tensor* ameva_repacked_base(const struct ggml_tensor* t);

// ---- graph engine (ggml-ameva.cpp) ----
// An ameva_qmv kernel exists for every work shape of this weight type.
bool ameva_qmv_has_type(enum ggml_type type);
AcContext* ameva_get_ac_context();
uint64_t ameva_max_buffer_size(void);

// ---- counters (ggml-ameva-device.cpp) ----
void ameva_count_repacked_tensor(uint64_t bytes);
void ameva_count_qmv_mul_mat(int64_t n);
void ameva_count_mmq_mul_mat(int64_t n);
void ameva_count_mmqb_mul_mat(int64_t n);
void ameva_count_mul_mat(int64_t n);
