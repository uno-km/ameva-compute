// SPDX-License-Identifier: Apache-2.0
// Write a GGUF model with the requested weight types through libllama's own quantizer (llama_model_quantize), so the
// AMEVA type matrix runs on real models on devices that have libllama but no llama-quantize binary.
//
//   ameva_quantize IN.gguf OUT.gguf TYPE [--pure] [--threads N]
//
// TYPE is a llama_ftype name (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q2_K_S, Q3_K_S, Q3_K_M, Q3_K_L,
// Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, IQ4_NL, IQ4_XS). --pure gives every tensor the base type wherever its row
// length allows it (llama.cpp otherwise mixes types per tensor role). Already quantized inputs are requantized: the
// result is a valid model of the requested types whose quality is bounded by the input, which is irrelevant when
// the model is used to compare two backends on identical weights.
//
// Build against the llama.h of the libllama that is loaded at run time (llama.cpp b8471 on the fleet): the
// parameter struct is passed by pointer but its layout differs between builds.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static const struct {
    const char* name;
    enum llama_ftype ftype;
} kTypes[] = {
    {"F32", LLAMA_FTYPE_ALL_F32},        {"F16", LLAMA_FTYPE_MOSTLY_F16},       {"BF16", LLAMA_FTYPE_MOSTLY_BF16},
    {"Q4_0", LLAMA_FTYPE_MOSTLY_Q4_0},   {"Q4_1", LLAMA_FTYPE_MOSTLY_Q4_1},     {"Q5_0", LLAMA_FTYPE_MOSTLY_Q5_0},
    {"Q5_1", LLAMA_FTYPE_MOSTLY_Q5_1},   {"Q8_0", LLAMA_FTYPE_MOSTLY_Q8_0},     {"Q2_K", LLAMA_FTYPE_MOSTLY_Q2_K},
    {"Q2_K_S", LLAMA_FTYPE_MOSTLY_Q2_K_S}, {"Q3_K_S", LLAMA_FTYPE_MOSTLY_Q3_K_S}, {"Q3_K_M", LLAMA_FTYPE_MOSTLY_Q3_K_M},
    {"Q3_K_L", LLAMA_FTYPE_MOSTLY_Q3_K_L}, {"Q4_K_S", LLAMA_FTYPE_MOSTLY_Q4_K_S}, {"Q4_K_M", LLAMA_FTYPE_MOSTLY_Q4_K_M},
    {"Q5_K_S", LLAMA_FTYPE_MOSTLY_Q5_K_S}, {"Q5_K_M", LLAMA_FTYPE_MOSTLY_Q5_K_M}, {"Q6_K", LLAMA_FTYPE_MOSTLY_Q6_K},
    {"IQ4_NL", LLAMA_FTYPE_MOSTLY_IQ4_NL}, {"IQ4_XS", LLAMA_FTYPE_MOSTLY_IQ4_XS},
};

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s IN.gguf OUT.gguf TYPE [--pure] [--threads N]\n", argv[0]);
        return 2;
    }
    const char* in = argv[1];
    const char* out = argv[2];
    const char* tname = argv[3];
    bool pure = false;
    int threads = 4;
    for (int i = 4; i < argc; ++i) {
        if (!strcmp(argv[i], "--pure")) {
            pure = true;
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    int found = -1;
    for (int i = 0; i < (int)(sizeof(kTypes) / sizeof(kTypes[0])); ++i) {
        if (!strcmp(kTypes[i].name, tname)) found = i;
    }
    if (found < 0) {
        fprintf(stderr, "unknown TYPE %s\n", tname);
        return 2;
    }

    llama_model_quantize_params qp = llama_model_quantize_default_params();
    qp.ftype = kTypes[found].ftype;
    qp.nthread = threads;
    qp.allow_requantize = true;
    qp.pure = pure;

    llama_backend_init();
    const uint32_t rc = llama_model_quantize(in, out, &qp);
    llama_backend_free();
    if (rc != 0) {
        fprintf(stderr, "llama_model_quantize(%s -> %s, %s) failed with %u\n", in, out, tname, rc);
        return 1;
    }
    printf("AMEVA_QUANTIZE=OK in=%s out=%s type=%s pure=%d\n", in, out, tname, (int)pure);
    return 0;
}
