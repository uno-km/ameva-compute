// SPDX-License-Identifier: Apache-2.0
//
// Print the devices a ggml host registers after ggml_backend_load_all(), which also loads the plugin named by
// GGML_BACKEND_PATH. For hosts without a --list-devices option (whisper-cli): build_ggml_ameva.sh needs a program
// that shows whether AMEVA0 registered against exactly this ggml.
//
//   clang -I <ggml>/include ggml_list_devices.c <libdir>/libggml.so.0 <libdir>/libggml-base.so.0 -o ggml_list_devices
//
// Output: one line per device, "<index> <type> <name> | <description>", GPU index in the order whisper.cpp counts
// it for -dev N (GPU and IGPU devices only).
#include <stdio.h>

#include "ggml-backend.h"

int main(void) {
    ggml_backend_load_all();
    size_t n = ggml_backend_dev_count();
    int gpu = 0;
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        const char *t = type == GGML_BACKEND_DEVICE_TYPE_CPU ? "CPU" : type == GGML_BACKEND_DEVICE_TYPE_GPU ? "GPU" :
                        type == GGML_BACKEND_DEVICE_TYPE_ACCEL ? "ACCEL" : "IGPU";
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            printf("%zu %s(-dev %d) %s | %s\n", i, t, gpu++, ggml_backend_dev_name(dev),
                   ggml_backend_dev_description(dev));
        } else {
            printf("%zu %s %s | %s\n", i, t, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        }
    }
    return n == 0;
}
