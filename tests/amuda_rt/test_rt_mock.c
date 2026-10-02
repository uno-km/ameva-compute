/* SPDX-License-Identifier: Apache-2.0 */
/* Host-plumbing tests for amuda_rt against the ac_runtime TEST DOUBLE (no GPU, nothing executes). */
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "amuda_rt.h"
#include "mock_ac_runtime.h"
#include "mock_images.h"

static int g_fail = 0;
#define EXPECT(cond) do { if (!(cond)) { printf("FAIL %s:%d: %s | detail: %s\n", __FILE__, __LINE__, #cond, amuGetLastErrorDetail()); g_fail++; } } while (0)

int main(void) {
    amuContext* ctx = NULL;
    EXPECT(amuInit(AMU_BACKEND_VULKAN, &ctx) == AMU_SUCCESS);
    EXPECT(strstr(amuGetDeviceName(ctx), "MOCK") != NULL);
    EXPECT(amuInit(AMU_BACKEND_OPENCL, &(amuContext*){0}) == AMU_ERR_BACKEND_UNAVAILABLE); /* not compiled in: no fallback */

    amuBuffer *a, *b, *c;
    EXPECT(amuMalloc(ctx, 4000, &a) == AMU_SUCCESS);
    EXPECT(amuMalloc(ctx, 4000, &b) == AMU_SUCCESS);
    EXPECT(amuMalloc(ctx, 4000, &c) == AMU_SUCCESS);
    EXPECT(amuMalloc(ctx, 0, &(amuBuffer*){0}) == AMU_ERR_INVALID_ARGUMENT);

    amuFunction* fn;
    EXPECT(amu_load_vector_add(ctx, &fn) == AMU_SUCCESS);
    EXPECT(amu_launch_vector_add(fn, amuDim(4, 1, 1), amuDim(256, 1, 1), a, b, c, 1000) == AMU_SUCCESS);
    EXPECT(g_mock.kernels_created == 1);
    EXPECT(g_mock.last_local[0] == 256 && g_mock.last_local[1] == 1 && g_mock.last_local[2] == 1);
    EXPECT(g_mock.last_spec[0] == 256 && g_mock.last_spec[1] == 1 && g_mock.last_spec[2] == 1);
    EXPECT(g_mock.last_nbuf == 3);
    EXPECT(g_mock.last_buf_mem[0] == amuHostPtr(a) && g_mock.last_buf_mem[1] == amuHostPtr(b) && g_mock.last_buf_mem[2] == amuHostPtr(c));
    int32_t n_seen; memcpy(&n_seen, g_mock.last_pc, 4);
    EXPECT(g_mock.last_pc_size == 4 && n_seen == 1000);
    EXPECT(g_mock.last_grid[0] == 4 && g_mock.last_grid[1] == 1 && g_mock.last_grid[2] == 1);
    EXPECT(amuDeviceSynchronize(ctx) == AMU_SUCCESS);
    EXPECT(fabs(amuLastGpuTimeMs(ctx) - 0.0015) < 1e-12);
    EXPECT(g_mock.events_live == 0);

    /* same block -> pipeline reused; new block -> new pipeline */
    EXPECT(amu_launch_vector_add(fn, amuDim(4, 1, 1), amuDim(256, 1, 1), a, b, c, 1000) == AMU_SUCCESS);
    EXPECT(g_mock.kernels_created == 1);
    EXPECT(amu_launch_vector_add(fn, amuDim(8, 1, 1), amuDim(128, 1, 1), a, b, c, 1000) == AMU_SUCCESS);
    EXPECT(g_mock.kernels_created == 2);
    /* in-order: second launch waited for the first before dispatching */
    EXPECT(g_mock.events_live == 1);
    EXPECT(amuDeviceSynchronize(ctx) == AMU_SUCCESS);
    EXPECT(g_mock.events_live == 0);

    /* launch validation */
    void* args3[3] = { &a, &b, &c };
    EXPECT(amuLaunchKernel(fn, amuDim(1, 1, 1), amuDim(64, 1, 1), args3, 3) == AMU_ERR_INVALID_ARGUMENT);
    EXPECT(amu_launch_vector_add(fn, amuDim(0, 1, 1), amuDim(64, 1, 1), a, b, c, 1) == AMU_ERR_LAUNCH);
    EXPECT(amu_launch_vector_add(fn, amuDim(1, 1, 1), amuDim(2048, 1, 1), a, b, c, 1) == AMU_ERR_LAUNCH);
    EXPECT(amu_launch_vector_add(fn, amuDim(1, 1, 1), amuDim(64, 64, 1), a, b, c, 1) == AMU_ERR_LAUNCH);
    EXPECT(amu_launch_vector_add(fn, amuDim(1, 1, 1), amuDim(64, 1, 1), a, NULL, c, 1) == AMU_ERR_INVALID_ARGUMENT);
    amuContext* ctx2 = NULL;
    amuBuffer* foreign = NULL;
    EXPECT(amuInit(AMU_BACKEND_VULKAN, &ctx2) == AMU_SUCCESS);
    EXPECT(amuMalloc(ctx2, 4000, &foreign) == AMU_SUCCESS);
    EXPECT(amu_launch_vector_add(fn, amuDim(1, 1, 1), amuDim(64, 1, 1), a, foreign, c, 1) == AMU_ERR_INVALID_ARGUMENT);

    /* float scalar packing: saxpy(alpha, x, y, n) -> push constants {alpha, n} */
    amuFunction* fs;
    EXPECT(amu_load_saxpy(ctx, &fs) == AMU_SUCCESS);
    EXPECT(amu_launch_saxpy(fs, amuDim(2, 1, 1), amuDim(64, 1, 1), 1.75f, a, b, 77u) == AMU_SUCCESS);
    float alpha_seen; uint32_t n2; memcpy(&alpha_seen, g_mock.last_pc, 4); memcpy(&n2, g_mock.last_pc + 4, 4);
    EXPECT(g_mock.last_pc_size == 8 && alpha_seen == 1.75f && n2 == 77u);
    EXPECT(g_mock.last_buf_mem[0] == amuHostPtr(a) && g_mock.last_buf_mem[1] == amuHostPtr(b));
    EXPECT(amuDeviceSynchronize(ctx) == AMU_SUCCESS);

    /* non-coherent memory: flush before dispatch, invalidate after wait */
    g_mock.non_coherent = 1;
    amuBuffer *x, *y;
    EXPECT(amuMalloc(ctx, 256, &x) == AMU_SUCCESS);
    EXPECT(amuMalloc(ctx, 256, &y) == AMU_SUCCESS);
    g_mock.nseq = 0;
    EXPECT(amu_launch_saxpy(fs, amuDim(1, 1, 1), amuDim(64, 1, 1), 2.0f, x, y, 64u) == AMU_SUCCESS);
    EXPECT(amuDeviceSynchronize(ctx) == AMU_SUCCESS);
    EXPECT(g_mock.nseq == 6 && memcmp(g_mock.seq, "FFDWII", 6) == 0);
    g_mock.non_coherent = 0;

    /* dispatch failure is reported, nothing left in flight */
    g_mock.fail_dispatch = 1;
    EXPECT(amu_launch_saxpy(fs, amuDim(1, 1, 1), amuDim(64, 1, 1), 2.0f, a, b, 64u) == AMU_ERR_LAUNCH);
    g_mock.fail_dispatch = 0;
    EXPECT(amuDeviceSynchronize(ctx) == AMU_SUCCESS);

    /* image without Vulkan code is rejected explicitly */
    amuKernelImage no_spv = amu_image_vector_add;
    no_spv.spirv = NULL;
    amuFunction* bad;
    EXPECT(amuModuleLoad(ctx, &no_spv, &bad) == AMU_ERR_KERNEL_IMAGE);
    amuKernelImage big_shared = amu_image_vector_add;
    big_shared.shared_bytes = 20000;
    EXPECT(amuModuleLoad(ctx, &big_shared, &bad) == AMU_ERR_NOT_SUPPORTED);
    amuKernelImage bad_abi = amu_image_vector_add;
    bad_abi.abi = 99;
    EXPECT(amuModuleLoad(ctx, &bad_abi, &bad) == AMU_ERR_KERNEL_IMAGE);

    /* pipeline build failure surfaces as KERNEL_BUILD */
    g_mock.fail_kernel = 1;
    EXPECT(amu_launch_vector_add(fn, amuDim(1, 1, 1), amuDim(32, 1, 1), a, b, c, 1) == AMU_ERR_KERNEL_BUILD);
    g_mock.fail_kernel = 0;

    /* timeout poisons the context */
    EXPECT(amu_launch_vector_add(fn, amuDim(1, 1, 1), amuDim(256, 1, 1), a, b, c, 1) == AMU_SUCCESS);
    g_mock.timeout_next = 1;
    EXPECT(amuDeviceSynchronize(ctx) == AMU_ERR_TIMEOUT);
    EXPECT(amu_launch_vector_add(fn, amuDim(1, 1, 1), amuDim(256, 1, 1), a, b, c, 1) == AMU_ERR_TIMEOUT);
    EXPECT(amuMalloc(ctx, 16, &(amuBuffer*){0}) == AMU_ERR_TIMEOUT);

    /* env selection never falls back. putenv with static storage: bionic's setenv allocates strings that
     * unsetenv never frees, which LeakSanitizer would report against this test instead of the runtime */
    static char env_cuda[] = "AMUDA_BACKEND=cuda", env_opencl[] = "AMUDA_BACKEND=opencl";
    putenv(env_cuda);
    EXPECT(amuInitFromEnv(&(amuContext*){0}) == AMU_ERR_INVALID_ARGUMENT);
    putenv(env_opencl);
    EXPECT(amuInitFromEnv(&(amuContext*){0}) == AMU_ERR_BACKEND_UNAVAILABLE);
    unsetenv("AMUDA_BACKEND");

    /* ctx is poisoned (sticky timeout): teardown still releases everything */
    EXPECT(amuModuleUnload(fn) == AMU_SUCCESS);
    EXPECT(amuModuleUnload(fs) == AMU_SUCCESS);
    amuFree(a); amuFree(b); amuFree(c); amuFree(x); amuFree(y); amuFree(foreign);
    amuDestroy(ctx);
    EXPECT(amuDestroy(ctx2) == AMU_SUCCESS);

    /* teardown failures are reported, never swallowed */
    {
        amuContext* t = NULL;
        EXPECT(amuInit(AMU_BACKEND_VULKAN, &t) == AMU_SUCCESS);
        amuBuffer* tb = NULL;
        EXPECT(amuMalloc(t, 64, &tb) == AMU_SUCCESS);
        amuFunction* tf = NULL;
        EXPECT(amu_load_vector_add(t, &tf) == AMU_SUCCESS);
        EXPECT(amu_launch_vector_add(tf, amuDim(1, 1, 1), amuDim(64, 1, 1), tb, tb, tb, 16) == AMU_SUCCESS);
        EXPECT(amuDeviceSynchronize(t) == AMU_SUCCESS);
        g_mock.fail_buffer_destroy = 1;
        EXPECT(amuFree(tb) == AMU_ERR_INTERNAL);
        EXPECT(strstr(amuGetLastErrorDetail(), "buffer_destroy") != NULL);
        g_mock.fail_buffer_destroy = 0;
        g_mock.fail_kernel_destroy = 1;
        EXPECT(amuModuleUnload(tf) == AMU_ERR_INTERNAL);
        EXPECT(strstr(amuGetLastErrorDetail(), "kernel_destroy") != NULL);
        g_mock.fail_kernel_destroy = 0;
        EXPECT(amuDestroy(t) == AMU_SUCCESS);
        EXPECT(amuDestroy(NULL) == AMU_SUCCESS);
        EXPECT(amuModuleUnload(NULL) == AMU_SUCCESS);
    }
    EXPECT(g_mock.buffers_created == g_mock.buffers_destroyed);
    EXPECT(g_mock.kernels_created == g_mock.kernels_destroyed);

    printf(g_fail ? "amuda_rt mock tests: %d FAILED\n" : "amuda_rt mock tests: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
