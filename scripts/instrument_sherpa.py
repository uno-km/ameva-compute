import base64
import subprocess

patch_code = '''
import re

impl_path = "/data/data/com.termux/files/home/sherpa-ncnn/sherpa-ncnn/csrc/offline-tts-vits-impl.h"
model_path = "/data/data/com.termux/files/home/sherpa-ncnn/sherpa-ncnn/csrc/offline-tts-vits-model.cc"

# 1. Patch offline-tts-vits-impl.h
with open(impl_path, "r") as f:
    impl_src = f.read()

# Backup if not already backed up
with open(impl_path + ".bak", "w") as f:
    f.write(impl_src)

# We need to instrument Process and Generate
# In Generate, record t_text and stage times
old_process = """    auto encoder_out = model_->RunEncoder(sequence);
    sequence.release();

    ncnn::Mat noise(encoder_out[0].w, 2);
    RandomVectorFill(static_cast<float *>(noise), noise.w * noise.h, 0,
                     noise_scale_w);

    ncnn::Mat g = model_->RunEmbedding(sid);

    ncnn::Mat logw = model_->RunDurationPredictor(encoder_out[0], noise, g);

    noise.release();
    encoder_out[0].release();

    ncnn::Mat z_p = model_->PathAttention(logw, encoder_out[1], encoder_out[2],
                                          noise_scale, speed);
    encoder_out.clear();
    logw.release();

    ncnn::Mat z = model_->RunFlow(z_p, g);
    z_p.release();

    ncnn::Mat o = model_->RunDecoder(z, g);"""

new_process = """    auto t_enc0 = std::chrono::high_resolution_clock::now();
    auto encoder_out = model_->RunEncoder(sequence);
    auto t_enc1 = std::chrono::high_resolution_clock::now();
    sequence.release();

    ncnn::Mat noise(encoder_out[0].w, 2);
    RandomVectorFill(static_cast<float *>(noise), noise.w * noise.h, 0,
                     noise_scale_w);

    ncnn::Mat g = model_->RunEmbedding(sid);

    auto t_dp0 = std::chrono::high_resolution_clock::now();
    ncnn::Mat logw = model_->RunDurationPredictor(encoder_out[0], noise, g);
    auto t_dp1 = std::chrono::high_resolution_clock::now();

    noise.release();
    encoder_out[0].release();

    auto t_pa0 = std::chrono::high_resolution_clock::now();
    ncnn::Mat z_p = model_->PathAttention(logw, encoder_out[1], encoder_out[2],
                                          noise_scale, speed);
    auto t_pa1 = std::chrono::high_resolution_clock::now();
    encoder_out.clear();
    logw.release();

    auto t_fl0 = std::chrono::high_resolution_clock::now();
    ncnn::Mat z = model_->RunFlow(z_p, g);
    auto t_fl1 = std::chrono::high_resolution_clock::now();
    z_p.release();

    auto t_dec0 = std::chrono::high_resolution_clock::now();
    ncnn::Mat o = model_->RunDecoder(z, g);
    auto t_dec1 = std::chrono::high_resolution_clock::now();

    double enc_ms = std::chrono::duration<double, std::milli>(t_enc1 - t_enc0).count();
    double dp_ms = std::chrono::duration<double, std::milli>(t_dp1 - t_dp0).count();
    double pa_ms = std::chrono::duration<double, std::milli>(t_pa1 - t_pa0).count();
    double fl_ms = std::chrono::duration<double, std::milli>(t_fl1 - t_fl0).count();
    double dec_ms = std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count();

    fprintf(stderr, "\\n----------------------------------------------------------------------\\n");
    fprintf(stderr, "[AMEVA-TTS-STAGE-TIMINGS]\\n");
    fprintf(stderr, "  ENCODER_TIME_MS=%.3f\\n", enc_ms);
    fprintf(stderr, "  DURATION_PREDICTOR_TIME_MS=%.3f\\n", dp_ms);
    fprintf(stderr, "  PATH_ATTENTION_TIME_MS=%.3f\\n", pa_ms);
    fprintf(stderr, "  FLOW_TIME_MS=%.3f\\n", fl_ms);
    fprintf(stderr, "  DECODER_TIME_MS=%.3f\\n", dec_ms);
    fprintf(stderr, "  NEURAL_STAGES_TOTAL_MS=%.3f\\n", enc_ms + dp_ms + pa_ms + fl_ms + dec_ms);
    fprintf(stderr, "----------------------------------------------------------------------\\n");"""

if old_process in impl_src:
    impl_src = impl_src.replace(old_process, new_process)
    print("SUCCESS: Patched Process() in offline-tts-vits-impl.h")
else:
    print("WARNING: Could not find exact old_process in offline-tts-vits-impl.h")

# Also patch text processing in Generate()
old_gen_convert = """    if (!args.text.empty()) {
      args.tokens = Convert(args.text);
    }"""

new_gen_convert = """    double text_proc_ms = 0.0;
    if (!args.text.empty()) {
      auto t_text0 = std::chrono::high_resolution_clock::now();
      args.tokens = Convert(args.text);
      auto t_text1 = std::chrono::high_resolution_clock::now();
      text_proc_ms = std::chrono::duration<double, std::milli>(t_text1 - t_text0).count();
      fprintf(stderr, "[AMEVA-TTS-STAGE-TIMINGS] TEXT_PROCESSING_TIME_MS=%.3f\\n", text_proc_ms);
    }"""

if old_gen_convert in impl_src:
    impl_src = impl_src.replace(old_gen_convert, new_gen_convert)
    print("SUCCESS: Patched Convert() in offline-tts-vits-impl.h")

# Add chrono include if missing
if "#include <chrono>" not in impl_src:
    impl_src = "#include <chrono>\\n" + impl_src

with open(impl_path, "w") as f:
    f.write(impl_src)

# 2. Patch offline-tts-vits-model.cc for MODEL_LOAD_TIME_MS
with open(model_path, "r") as f:
    model_src = f.read()

with open(model_path + ".bak", "w") as f:
    f.write(model_src)

old_init = """  void Init() {
    meta_ = ReadFromConfigJson(config_.vits.model_dir + "/config.json");
    InitNet();
  }"""

new_init = """  void Init() {
    auto t_init0 = std::chrono::high_resolution_clock::now();
    meta_ = ReadFromConfigJson(config_.vits.model_dir + "/config.json");
    InitNet();
    auto t_init1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();
    fprintf(stderr, "[AMEVA-TTS-STAGE-TIMINGS] MODEL_LOAD_TIME_MS=%.3f\\n", load_ms);
  }"""

if old_init in model_src:
    model_src = model_src.replace(old_init, new_init)
    print("SUCCESS: Patched Init() in offline-tts-vits-model.cc")

if "#include <chrono>" not in model_src:
    model_src = "#include <chrono>\\n" + model_src

with open(model_path, "w") as f:
    f.write(model_src)

print("INSTRUMENTATION_COMPLETE")
'''

b = base64.b64encode(patch_code.encode()).decode()
cmd = f"echo {b} | base64 -d | python3"

res = subprocess.run(["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21", cmd], capture_output=True, text=True)
print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)
print("RETURNCODE:", res.returncode)
