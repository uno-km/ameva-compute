/**
 * @file ac_error.cpp
 * @brief AMUDA Generation-Based Error Tracking Implementation
 * Owned by AMEVA Compute Runtime Core.
 * Provides unified cross-thread error state propagation and generation lifecycle.
 */

#include "ac_error.h"
#include <mutex>
#include <unordered_map>

static std::mutex s_error_mutex;
static std::unordered_map<uint64_t, AmevaBackendErrorState> s_context_errors;

static AmevaBackendErrorState& get_or_create_state(uint64_t context_id) {
    auto it = s_context_errors.find(context_id);
    if (it == s_context_errors.end()) {
        AmevaBackendErrorState init = {1, 0, AMEVA_BACKEND_OP_NONE, 0, 0, 0};
        auto res = s_context_errors.emplace(context_id, init);
        return res.first->second;
    }
    return it->second;
}

/* Multi-Context Error Tracking Implementation */
void ac_runtime_set_error_ctx(
    uint64_t context_id,
    int32_t error_code,
    AmevaBackendOpKind op,
    size_t offset,
    size_t requested_bytes,
    size_t capacity_bytes
) {
    if (error_code == 0) return;
    std::lock_guard<std::mutex> lock(s_error_mutex);
    AmevaBackendErrorState& state = get_or_create_state(context_id);
    // First-error-wins within generation
    if (state.first_error == 0) {
        state.first_error = error_code;
        state.operation = op;
        state.offset = offset;
        state.requested_bytes = requested_bytes;
        state.capacity_bytes = capacity_bytes;
    }
}

AmevaBackendErrorState ac_runtime_get_error_state_ctx(uint64_t context_id) {
    std::lock_guard<std::mutex> lock(s_error_mutex);
    return get_or_create_state(context_id);
}

int32_t ac_runtime_consume_error_ctx(uint64_t context_id) {
    std::lock_guard<std::mutex> lock(s_error_mutex);
    AmevaBackendErrorState& state = get_or_create_state(context_id);
    int32_t err = state.first_error;
    state.first_error = 0;
    state.operation = AMEVA_BACKEND_OP_NONE;
    state.offset = 0;
    state.requested_bytes = 0;
    state.capacity_bytes = 0;
    return err;
}

uint64_t ac_runtime_advance_generation_ctx(uint64_t context_id) {
    std::lock_guard<std::mutex> lock(s_error_mutex);
    AmevaBackendErrorState& state = get_or_create_state(context_id);
    state.generation++;
    state.first_error = 0;
    state.operation = AMEVA_BACKEND_OP_NONE;
    state.offset = 0;
    state.requested_bytes = 0;
    state.capacity_bytes = 0;
    return state.generation;
}

uint64_t ac_runtime_get_generation_ctx(uint64_t context_id) {
    std::lock_guard<std::mutex> lock(s_error_mutex);
    return get_or_create_state(context_id).generation;
}

void ac_runtime_reset_error_state_ctx(uint64_t context_id) {
    std::lock_guard<std::mutex> lock(s_error_mutex);
    AmevaBackendErrorState& state = get_or_create_state(context_id);
    state.first_error = 0;
    state.operation = AMEVA_BACKEND_OP_NONE;
    state.offset = 0;
    state.requested_bytes = 0;
    state.capacity_bytes = 0;
}

/* Default / Global Context Overloads */
void ac_runtime_set_error(
    int32_t error_code,
    AmevaBackendOpKind op,
    size_t offset,
    size_t requested_bytes,
    size_t capacity_bytes
) {
    ac_runtime_set_error_ctx(AC_GLOBAL_CONTEXT_ID, error_code, op, offset, requested_bytes, capacity_bytes);
}

AmevaBackendErrorState ac_runtime_get_error_state(void) {
    return ac_runtime_get_error_state_ctx(AC_GLOBAL_CONTEXT_ID);
}

int32_t ac_runtime_consume_error(void) {
    return ac_runtime_consume_error_ctx(AC_GLOBAL_CONTEXT_ID);
}

uint64_t ac_runtime_advance_generation(void) {
    return ac_runtime_advance_generation_ctx(AC_GLOBAL_CONTEXT_ID);
}

uint64_t ac_runtime_get_generation(void) {
    return ac_runtime_get_generation_ctx(AC_GLOBAL_CONTEXT_ID);
}

void ac_runtime_reset_error_state(void) {
    ac_runtime_reset_error_state_ctx(AC_GLOBAL_CONTEXT_ID);
}
