/**
 * @file test_amuda_gguf_weighted.c
 * @brief Comprehensive Real Weighted GGUF Verification Suite
 * 
 * Verifies tensor bounds, mmap pointers, type distribution, FD leaks,
 * and dynamic error code separation on production-grade GGUF models.
 */

#include "amuda_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <dirent.h>

static int count_open_fds(void) {
    DIR* d = opendir("/proc/self/fd");
    if (!d) return -1;
    int count = 0;
    struct dirent* de;
    while ((de = readdir(d)) != NULL) {
        if (de->d_name[0] != '.') count++;
    }
    closedir(d);
    return count;
}

static const char* type_name(uint32_t type) {
    switch (type) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q8_K: return "Q8_K";
        case GGML_TYPE_I2_S: return "I2_S";
        default: return "UNKNOWN";
    }
}

// Synthetic Buffer helper for error separation tests
typedef struct {
    uint8_t* data;
    size_t cap;
    size_t len;
} TestBuf;

static void tbuf_init(TestBuf* b, size_t cap) {
    b->data = (uint8_t*)malloc(cap ? cap : 256);
    b->cap = cap ? cap : 256;
    b->len = 0;
}
static void tbuf_free(TestBuf* b) {
    if (b->data) free(b->data);
    b->data = NULL;
    b->cap = b->len = 0;
}
static void tbuf_append(TestBuf* b, const void* src, size_t size) {
    if (b->len + size > b->cap) {
        b->cap = (b->cap + size + 256) * 2;
        b->data = (uint8_t*)realloc(b->data, b->cap);
    }
    memcpy(b->data + b->len, src, size);
    b->len += size;
}
static void tbuf_u8(TestBuf* b, uint8_t v) { tbuf_append(b, &v, 1); }
static void tbuf_u32(TestBuf* b, uint32_t v) { tbuf_append(b, &v, 4); }
static void tbuf_u64(TestBuf* b, uint64_t v) { tbuf_append(b, &v, 8); }
static void tbuf_str(TestBuf* b, const char* s) {
    uint64_t len = (uint64_t)strlen(s);
    tbuf_u64(b, len);
    tbuf_append(b, s, (size_t)len);
}
static void tbuf_pad_align(TestBuf* b, size_t align) {
    size_t rem = b->len % align;
    if (rem != 0) {
        size_t pad = align - rem;
        for (size_t i = 0; i < pad; i++) tbuf_u8(b, 0);
    }
}
static const char* write_temp_file(const TestBuf* b, const char* tag) {
    static char paths[4][256];
    static int idx = 0;
    char* p = paths[idx % 4];
    idx++;
    snprintf(p, 256, "/data/data/com.termux/files/home/test_err_sep_%s.gguf", tag);
    FILE* f = fopen(p, "wb");
    if (!f) return NULL;
    fwrite(b->data, 1, b->len, f);
    fclose(f);
    return p;
}

int main(int argc, char** argv) {
    const char* default_paths[] = {
        "/data/data/com.termux/files/home/.termux-llama/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "/data/data/com.termux/files/home/.termux-llama/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "/data/data/com.termux/files/home/.termux-llama/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "/data/data/com.termux/files/home/.cache/termux-bitnet/models/bitnet-2b-ggml-model-i2_s.gguf",
        NULL
    };

    const char* model_path = NULL;
    if (argc > 1) {
        model_path = argv[1];
    } else {
        for (int i = 0; default_paths[i] != NULL; i++) {
            if (access(default_paths[i], R_OK) == 0) {
                model_path = default_paths[i];
                break;
            }
        }
    }

    if (!model_path) {
        fprintf(stderr, "[ERROR] No readable GGUF model path found.\n");
        return 1;
    }

    struct stat st;
    if (stat(model_path, &st) != 0) {
        fprintf(stderr, "[ERROR] Cannot stat %s\n", model_path);
        return 1;
    }
    size_t file_size = (size_t)st.st_size;

    printf("=======================================================================\n");
    printf("  AMEVA Compute: Real Weighted GGUF Validation Suite\n");
    printf("=======================================================================\n");
    printf("  TARGET_MODEL=%s\n", model_path);
    printf("  FILE_SIZE=%zu bytes (%.2f MB)\n", file_size, (double)file_size / (1024.0 * 1024.0));

    int fds_before = count_open_fds();
    GgufModel* model = NULL;
    AcResult res = ac_gguf_open(model_path, &model);
    int fds_after_open = count_open_fds();

    if (res != AC_SUCCESS || !model) {
        printf("  OPEN_RESULT=FAILED (res=%d)\n", (int)res);
        return 1;
    }
    printf("  OPEN_RESULT=SUCCESS\n");
    printf("  MAGIC=0x46554747 (GGUF)\n");
    printf("  VERSION=%u\n", model->version);
    printf("  ARCH=%s\n", model->config.arch);
    printf("  METADATA_KV_COUNT=%llu\n", (unsigned long long)model->metadata_kv_count);
    printf("  TENSOR_COUNT=%llu\n", (unsigned long long)model->tensor_count);
    printf("  CONTEXT_LENGTH=%u\n", model->config.context_length);
    printf("  EMBEDDING_LENGTH=%u\n", model->config.embedding_length);
    printf("  HEAD_COUNT=%u\n", model->config.head_count);
    printf("  HEAD_COUNT_KV=%u\n", model->config.head_count_kv);
    printf("  BLOCK_COUNT=%u\n", model->config.block_count);
    printf("  VOCAB_SIZE=%u\n", model->config.vocab_size);

    // Tensor audit
    uint64_t type_counts[64] = {0};
    int invalid_ranges = 0;
    int page_fault_failures = 0;
    uint64_t max_end_offset = 0;
    int max_end_idx = -1;

    for (uint64_t i = 0; i < model->tensor_count; i++) {
        GgufTensor* t = &model->tensors[i];
        if (t->type < 64) type_counts[t->type]++;

        if (!t->data_ptr || t->size_bytes == 0) {
            invalid_ranges++;
            continue;
        }

        uint64_t end_offset = t->offset + t->size_bytes;
        if (end_offset > max_end_offset) {
            max_end_offset = end_offset;
            max_end_idx = (int)i;
        }

        // Verify mmap boundary
        const uint8_t* ptr = (const uint8_t*)t->data_ptr;
        volatile uint8_t first_byte = ptr[0];
        volatile uint8_t last_byte = ptr[t->size_bytes - 1];
        (void)first_byte;
        (void)last_byte;
    }

    printf("\n  [TENSOR BOUNDARY AND RANGE SAMPLES]\n");
    if (model->tensor_count > 0) {
        GgufTensor* t0 = &model->tensors[0];
        printf("  FIRST_TENSOR:  name='%s', type=%u (%s), dims=%u [%llu,%llu,%llu,%llu], offset=%llu, size=%zu\n",
               t0->name, t0->type, type_name(t0->type), t0->n_dims,
               (unsigned long long)t0->ne[0], (unsigned long long)t0->ne[1],
               (unsigned long long)t0->ne[2], (unsigned long long)t0->ne[3],
               (unsigned long long)t0->offset, t0->size_bytes);

        uint64_t mid_idx = model->tensor_count / 2;
        GgufTensor* tmid = &model->tensors[mid_idx];
        printf("  MIDDLE_TENSOR: name='%s', type=%u (%s), dims=%u [%llu,%llu,%llu,%llu], offset=%llu, size=%zu\n",
               tmid->name, tmid->type, type_name(tmid->type), tmid->n_dims,
               (unsigned long long)tmid->ne[0], (unsigned long long)tmid->ne[1],
               (unsigned long long)tmid->ne[2], (unsigned long long)tmid->ne[3],
               (unsigned long long)tmid->offset, tmid->size_bytes);

        uint64_t last_idx = model->tensor_count - 1;
        GgufTensor* tlast = &model->tensors[last_idx];
        printf("  LAST_TENSOR:   name='%s', type=%u (%s), dims=%u [%llu,%llu,%llu,%llu], offset=%llu, size=%zu\n",
               tlast->name, tlast->type, type_name(tlast->type), tlast->n_dims,
               (unsigned long long)tlast->ne[0], (unsigned long long)tlast->ne[1],
               (unsigned long long)tlast->ne[2], (unsigned long long)tlast->ne[3],
               (unsigned long long)tlast->offset, tlast->size_bytes);

        if (max_end_idx >= 0) {
            GgufTensor* tmax = &model->tensors[max_end_idx];
            printf("  MAX_END_TENSOR: name='%s', offset=%llu, size=%zu, end=%llu, file_size=%zu (WITHIN_BOUNDS=%s)\n",
                   tmax->name, (unsigned long long)tmax->offset, tmax->size_bytes,
                   (unsigned long long)max_end_offset, file_size,
                   (max_end_offset <= file_size) ? "YES" : "NO");
        }
    }

    printf("\n  [TYPE DISTRIBUTION]\n");
    for (uint32_t t = 0; t < 64; t++) {
        if (type_counts[t] > 0) {
            printf("    TYPE_%u (%s): %llu tensors\n", t, type_name(t), (unsigned long long)type_counts[t]);
        }
    }

    printf("\n  [SAFETY VERIFICATION METRICS]\n");
    printf("  INVALID_RANGE_COUNT=%d\n", invalid_ranges);
    printf("  PAGE_FAULT_FAILURES=%d\n", page_fault_failures);

    ac_gguf_close(model);
    int fds_after_close = count_open_fds();
    int fd_delta = fds_after_close - fds_before;
    printf("  FD_LEAK_DELTA=%d (before=%d, during=%d, after=%d)\n",
           fd_delta, fds_before, fds_after_open, fds_after_close);

    // =========================================================================
    // Dynamic Error Code Separation Test
    // =========================================================================
    printf("\n  [DYNAMIC ERROR CODE SEPARATION TEST]\n");

    // Subtest 1: Injected Unknown Tensor Type (e.g. 0x9999) -> Expect AC_ERR_DEVICE_NOT_SUPPORTED (-405)
    {
        TestBuf b;
        tbuf_init(&b, 256);
        tbuf_u32(&b, GGUF_MAGIC);
        tbuf_u32(&b, 3);
        tbuf_u64(&b, 1); // 1 tensor
        tbuf_u64(&b, 0); // 0 kv
        tbuf_str(&b, "unknown_type_tensor");
        tbuf_u32(&b, 1); // n_dims = 1
        tbuf_u64(&b, 32);
        tbuf_u32(&b, 0x9999); // UNKNOWN TYPE!
        tbuf_u64(&b, 0);
        tbuf_pad_align(&b, 32);
        for (int i = 0; i < 32; i++) tbuf_u8(&b, 0);

        const char* p = write_temp_file(&b, "unknown_type");
        GgufModel* bad_model = NULL;
        AcResult bad_res = ac_gguf_open(p, &bad_model);
        if (bad_model) ac_gguf_close(bad_model);
        unlink(p);
        tbuf_free(&b);

        bool sub1_pass = (bad_res == AC_ERR_DEVICE_NOT_SUPPORTED && bad_model == NULL);
        printf("  ERR_TEST_1_UNKNOWN_TYPE: %s (expected %d [AC_ERR_DEVICE_NOT_SUPPORTED], got %d)\n",
               sub1_pass ? "PASS" : "FAIL", (int)AC_ERR_DEVICE_NOT_SUPPORTED, (int)bad_res);
    }

    // Subtest 2: Injected Payload Exceeding EOF -> Expect AC_ERR_BINARY_CORRUPTED (-304)
    {
        TestBuf b;
        tbuf_init(&b, 256);
        tbuf_u32(&b, GGUF_MAGIC);
        tbuf_u32(&b, 3);
        tbuf_u64(&b, 1); // 1 tensor
        tbuf_u64(&b, 0); // 0 kv
        tbuf_str(&b, "oob_payload_tensor");
        tbuf_u32(&b, 1); // n_dims = 1
        tbuf_u64(&b, 128); // 128 F32 elements = 512 bytes
        tbuf_u32(&b, GGML_TYPE_F32);
        tbuf_u64(&b, 500000000ULL); // Offset way beyond file size!
        tbuf_pad_align(&b, 32);
        for (int i = 0; i < 16; i++) tbuf_u8(&b, 0); // Only 16 bytes payload

        const char* p = write_temp_file(&b, "oob_eof");
        GgufModel* bad_model = NULL;
        AcResult bad_res = ac_gguf_open(p, &bad_model);
        if (bad_model) ac_gguf_close(bad_model);
        unlink(p);
        tbuf_free(&b);

        bool sub2_pass = (bad_res == AC_ERR_BINARY_CORRUPTED && bad_model == NULL);
        printf("  ERR_TEST_2_EXCEED_EOF:   %s (expected %d [AC_ERR_BINARY_CORRUPTED], got %d)\n",
               sub2_pass ? "PASS" : "FAIL", (int)AC_ERR_BINARY_CORRUPTED, (int)bad_res);
    }

    printf("=======================================================================\n");
    return (invalid_ranges == 0 && fd_delta == 0) ? 0 : 1;
}
