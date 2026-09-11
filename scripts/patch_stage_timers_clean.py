import subprocess
import sys

base_cmd = ["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21"]

print("Fetching offline-tts-vits-impl.h from A35...")
res = subprocess.run(base_cmd + ["cat ~/sherpa-ncnn/sherpa-ncnn/csrc/offline-tts-vits-impl.h"], capture_output=True, text=True, encoding="utf-8", check=True)
impl_src = res.stdout

print("Fetching offline-tts-vits-model.cc from A35...")
res = subprocess.run(base_cmd + ["cat ~/sherpa-ncnn/sherpa-ncnn/csrc/offline-tts-vits-model.cc"], capture_output=True, text=True, encoding="utf-8", check=True)
model_src = res.stdout

# 1. Modify impl_src
if "#include <chrono>" not in impl_src:
    impl_src = "#include <chrono>\n" + impl_src

target_process = """    auto encoder_out = model_->RunEncoder(sequence);
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

replacement_process = """    auto t_enc0 = std::chrono::high_resolution_clock::now();
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
    fprintf(stderr, "  TOTAL_NEURAL_TIME_MS=%.3f\\n", enc_ms + dp_ms + pa_ms + fl_ms + dec_ms);
    fprintf(stderr, "----------------------------------------------------------------------\\n");"""

assert target_process in impl_src, "target_process not found in impl_src"
impl_src = impl_src.replace(target_process, replacement_process)

target_gen_start = """    if (!args.text.empty()) {
      args.tokens = Convert(args.text);
    }"""

replacement_gen_start = """    auto t_synth0 = std::chrono::high_resolution_clock::now();
    double text_proc_ms = 0.0;
    if (!args.text.empty()) {
      auto t_tx0 = std::chrono::high_resolution_clock::now();
      args.tokens = Convert(args.text);
      auto t_tx1 = std::chrono::high_resolution_clock::now();
      text_proc_ms = std::chrono::duration<double, std::milli>(t_tx1 - t_tx0).count();
      fprintf(stderr, "[AMEVA-TTS-STAGE-TIMINGS] TEXT_PROCESSING_TIME_MS=%.3f\\n", text_proc_ms);
    }"""

assert target_gen_start in impl_src, "target_gen_start not found in impl_src"
impl_src = impl_src.replace(target_gen_start, replacement_gen_start)

target_gen_end = """    GeneratedAudio ans;
    ans.sample_rate = meta_data.sample_rate;
    ans.samples = std::move(samples);

    return ans;"""

replacement_gen_end = """    auto t_synth1 = std::chrono::high_resolution_clock::now();
    double total_synth_ms = std::chrono::duration<double, std::milli>(t_synth1 - t_synth0).count();
    fprintf(stderr, "[AMEVA-TTS-STAGE-TIMINGS] TOTAL_SYNTHESIS_TIME_MS=%.3f\\n", total_synth_ms);

    GeneratedAudio ans;
    ans.sample_rate = meta_data.sample_rate;
    ans.samples = std::move(samples);

    return ans;"""

assert target_gen_end in impl_src, "target_gen_end not found in impl_src"
impl_src = impl_src.replace(target_gen_end, replacement_gen_end)

# 2. Modify model_src
if "#include <chrono>" not in model_src:
    model_src = "#include <chrono>\n" + model_src

target_init = """  void Init() {
    meta_ = ReadFromConfigJson(config_.vits.model_dir + "/config.json");
    InitNet();
  }"""

replacement_init = """  void Init() {
    auto t_ld0 = std::chrono::high_resolution_clock::now();
    meta_ = ReadFromConfigJson(config_.vits.model_dir + "/config.json");
    InitNet();
    auto t_ld1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_ld1 - t_ld0).count();
    fprintf(stderr, "[AMEVA-TTS-STAGE-TIMINGS] MODEL_LOAD_TIME_MS=%.3f\\n", load_ms);
  }"""

assert target_init in model_src, "target_init not found in model_src"
model_src = model_src.replace(target_init, replacement_init)

# Save modified files to local temp directory and tar them
import os, tarfile, http.server, socketserver, threading, time

os.makedirs("build_temp", exist_ok=True)
with open("build_temp/offline-tts-vits-impl.h", "w", encoding="utf-8", newline="\n") as f:
    f.write(impl_src)
with open("build_temp/offline-tts-vits-model.cc", "w", encoding="utf-8", newline="\n") as f:
    f.write(model_src)

tar_name = "sherpa_patch.tar.gz"
with tarfile.open(tar_name, "w:gz") as tar:
    tar.add("build_temp/offline-tts-vits-impl.h", arcname="offline-tts-vits-impl.h")
    tar.add("build_temp/offline-tts-vits-model.cc", arcname="offline-tts-vits-model.cc")

PORT = 8089
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

socketserver.TCPServer.allow_reuse_address = True
server = socketserver.TCPServer(("0.0.0.0", PORT), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
print("HTTP server running on port 8089...")
time.sleep(1)

cmd = (
    "curl -s -f http://100.101.241.92:8089/sherpa_patch.tar.gz -o ~/sherpa_patch.tar.gz && "
    "tar -xzf ~/sherpa_patch.tar.gz -C ~/sherpa-ncnn/sherpa-ncnn/csrc/ && "
    "rm -f ~/sherpa_patch.tar.gz && "
    "ninja -C ~/sherpa-ncnn/build-vulkan sherpa-ncnn-offline-tts && "
    "rm -f ~/sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts-mixed && "
    "cp ~/sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts ~/sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts-mixed && "
    "sha256sum ~/sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts-mixed"
)

res = subprocess.run(base_cmd + [cmd], capture_output=True, text=True, encoding="utf-8")
print("STDOUT:\n", res.stdout)
print("STDERR:\n", res.stderr)
print("RETURNCODE:", res.returncode)

server.shutdown()
if os.path.exists(tar_name):
    os.remove(tar_name)
import shutil
shutil.rmtree("build_temp", ignore_errors=True)

if res.returncode != 0:
    sys.exit(1)

print("STAGE_TIMERS_PATCH_AND_BUILD_SUCCESS")
