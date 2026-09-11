import subprocess
import os
import sys

base_cmd = ["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21"]

print("=== RUNNING NORMAL AMEVA MIXED TTS ON A35 ===")
# Remove previous output
subprocess.run(base_cmd + ["rm -f ~/wave1a_mixed_en1.wav"], check=True)

remote_cmd = (
    "cd ~/ && "
    "unset AMEVA_TTS_FAIL_STAGE && "
    "./sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts-mixed "
    "--vits-model-dir=./ncnn-vits-piper-en_US-lessac-high-fp16-patched "
    "--use-vulkan-compute=1 --num-threads=1 "
    "--output-filename=./wave1a_mixed_en1.wav "
    "'The AMEVA text to speech pipeline is running.'"
)

res = subprocess.run(base_cmd + [remote_cmd], capture_output=True, text=True)

print("--- STDOUT ---")
print(res.stdout)
print("--- STDERR ---")
print(res.stderr)
print(f"RETURNCODE: {res.returncode}")

with open("normal_mixed_tts_stdout.log", "w", encoding="utf-8") as f:
    f.write(res.stdout)
with open("normal_mixed_tts_stderr.log", "w", encoding="utf-8") as f:
    f.write(res.stderr)

if res.returncode != 0:
    print("ERROR: Execution failed!")
    sys.exit(1)

print("NORMAL_MIXED_TTS_RUN_COMPLETED")
