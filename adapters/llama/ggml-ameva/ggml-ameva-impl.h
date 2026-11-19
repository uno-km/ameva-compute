/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial
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
// Called by graph_compute for each MUL_MAT src0: repack a weight-buffer tensor uploaded before the buffer was marked
// GGML_BACKEND_BUFFER_USAGE_WEIGHTS. Idempotent; returns true if the tensor is (now) repacked.
bool ameva_ensure_weight_repacked(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor);

// ---- load profile (ggml-ameva-buffer.cpp) ----
// AMEVA_LOAD_PROFILE=1 times the model-load phases inside this backend and prints them at process exit as
// "AMEVA_LOAD ..." lines. =2 additionally touches every source page of a repacked upload before repacking it, so that
// the page faults of the source (file I/O on an mmap'ed model) are timed apart from the CPU repack. Off: no clock reads.
enum AmevaLoadPhase {
    AMEVA_LOAD_CONTEXT = 0,    // ac_runtime_context_create + limits + pipeline cache load
    AMEVA_LOAD_ALLOC,          // ac_runtime_buffer_create + get_address (vkAllocateMemory, map; driver zeroes pages)
    AMEVA_LOAD_SRC_TOUCH,      // =2 only: first touch of the source pages of repacked uploads
    AMEVA_LOAD_REPACK_CPU,     // ameva_repack_rows (includes source page faults unless =2)
    AMEVA_LOAD_MAPPED_WRITE,   // staged repacked rows written into the mapped buffer
    AMEVA_LOAD_MAPPED_READ,    // repacked rows read back (partial writes, late repack)
    AMEVA_LOAD_PLAIN_WRITE,    // set_tensor of tensors kept in the ggml layout
    AMEVA_LOAD_SET_REPACKED,   // set_tensor of repacked tensors, whole call (contains the three phases above)
    AMEVA_LOAD_LATE_REPACK,    // ameva_ensure_weight_repacked, whole call (whisper.cpp marks usage after upload)
    AMEVA_LOAD_CACHE_READ,     // set_tensor served from the repack cache file (AMEVA_REPACK_CACHE), whole call
    AMEVA_LOAD_NPHASES
};
bool ameva_load_profile_level(void);  // true when AMEVA_LOAD_PROFILE >= 1
uint64_t ameva_load_now_ns(void);      // 0 when the profile is off
void ameva_load_add(enum AmevaLoadPhase phase, uint64_t t0_ns, uint64_t bytes);
void ameva_load_mark_first_graph(void);
// The load is over (first graph): complete or report the offline repack cache (ggml-ameva-repack-cache.h).
void ameva_repack_cache_first_graph(void);

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
