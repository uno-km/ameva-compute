/* Aborting stubs for the stable-diffusion.cpp / ggml symbols that ameva_sd_api.cpp references but that
 * ameva_sd_ref_check never reaches. Reaching one means the test left the path it is meant to cover, so it
 * stops with the symbol name instead of returning a made-up value. (The calls the test does exercise are
 * recording test doubles in ameva_sd_ref_check.cpp.) Signatures are irrelevant at link time for C symbols. */
#include <stdio.h>
#include <stdlib.h>

#define UNREACHABLE_STUB(name)                                                         \
    void name(void) {                                                                  \
        fprintf(stderr, "ameva_sd_ref_check: unexpected call to " #name "\n");         \
        abort();                                                                       \
    }

UNREACHABLE_STUB(free_sd_audio)
UNREACHABLE_STUB(free_upscaler_ctx)
UNREACHABLE_STUB(generate_video)
UNREACHABLE_STUB(get_upscale_factor)
UNREACHABLE_STUB(ggml_backend_dev_memory)
UNREACHABLE_STUB(ggml_backend_dev_supports_op)
UNREACHABLE_STUB(ggml_backend_load)
UNREACHABLE_STUB(ggml_backend_reg_dev_count)
UNREACHABLE_STUB(ggml_backend_reg_dev_get)
UNREACHABLE_STUB(ggml_backend_reg_get_proc_address)
UNREACHABLE_STUB(ggml_conv_2d_direct)
UNREACHABLE_STUB(ggml_flash_attn_ext)
UNREACHABLE_STUB(ggml_flash_attn_ext_set_prec)
UNREACHABLE_STUB(ggml_free)
UNREACHABLE_STUB(ggml_init)
UNREACHABLE_STUB(ggml_new_tensor_4d)
UNREACHABLE_STUB(ggml_tensor_overhead)
UNREACHABLE_STUB(gguf_free)
UNREACHABLE_STUB(gguf_get_n_tensors)
UNREACHABLE_STUB(gguf_get_tensor_name)
UNREACHABLE_STUB(gguf_get_tensor_size)
UNREACHABLE_STUB(gguf_init_from_file)
UNREACHABLE_STUB(new_upscaler_ctx)
UNREACHABLE_STUB(preprocess_canny)
UNREACHABLE_STUB(sd_vid_gen_params_init)
UNREACHABLE_STUB(str_to_sample_method)
UNREACHABLE_STUB(str_to_scheduler)
UNREACHABLE_STUB(str_to_sd_hires_upscaler)
UNREACHABLE_STUB(upscale)
