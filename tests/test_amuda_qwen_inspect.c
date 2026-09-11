#include <stdio.h>
#include <string.h>
#include "amuda_gguf.h"

int main(int argc, char** argv) {
    const char* path = "/data/data/com.termux/files/home/.termux-llama/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    if (argc > 1) path = argv[1];

    GgufModel* model = NULL;
    AcResult res = ac_gguf_open(path, &model);
    if (res != AC_SUCCESS || !model) {
        printf("Failed to open %s (res=%d)\n", path, (int)res);
        return 1;
    }

    printf("MODEL: %s (tensors: %llu)\n", path, (unsigned long long)model->tensor_count);
    for (uint64_t i = 0; i < model->tensor_count; i++) {
        GgufTensor* t = &model->tensors[i];
        if (strstr(t->name, "blk.0.") || strstr(t->name, "output_norm") || strstr(t->name, "token_embd")) {
            printf("  TENSOR: %-32s type=%2u dims=%u [%llu, %llu, %llu] size=%zu\n",
                   t->name, t->type, t->n_dims,
                   (unsigned long long)t->ne[0],
                   (unsigned long long)t->ne[1],
                   (unsigned long long)t->ne[2],
                   t->size_bytes);
        }
    }

    ac_gguf_close(model);
    return 0;
}
