/**
 * Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 OR Commercial-License
 */
/* TEST DOUBLE state, see mock_ac_runtime.c */
#ifndef MOCK_AC_RUNTIME_H
#define MOCK_AC_RUNTIME_H
#include <stdint.h>
#define MOCK_SEQ 256
typedef struct {
    int fail_context, fail_kernel, fail_dispatch, timeout_next, non_coherent;
    int fail_buffer_destroy, fail_kernel_destroy; /* resources are still freed; only the status is an error */
    int buffers_created, buffers_destroyed, kernels_created, kernels_destroyed, dispatches, events_live;
    int flushes, invalidates;
    char seq[MOCK_SEQ]; int nseq;
    uint32_t last_local[3], last_spec[3], last_grid[3], last_nbuf, last_pc_size;
    void* last_buf_mem[16];
    unsigned char last_pc[128];
} MockLog;
extern MockLog g_mock;
#endif
