#!/usr/bin/env python3
import subprocess
import os
import re
import hashlib
import unicodedata

BIN = "/data/data/com.termux/files/home/ameva-work/stt/source/third_party/whisper.cpp/build-whisper-cpu/bin/whisper-cli"
MODEL_SMALL = "/data/data/com.termux/files/home/ameva-work/stt/models/ggml-small.bin"
AUDIO_EN = "/data/data/com.termux/files/home/ameva-work/stt/fixtures/jfk.wav"
AUDIO_KO = "/data/data/com.termux/files/home/ameva-work/stt/fixtures/korean/k02_ameva.wav"

REF_EN = "and so my fellow americans ask not what your country can do for you ask what you can do for your country"

env = os.environ.copy()
env["AMEVA_SPIRV_F16_PATH"] = "/data/data/com.termux/files/home/matmul_f16_f32.spv"
env["AMEVA_SPIRV_Q5_0_PATH"] = "/data/data/com.termux/files/home/matmul_q5_0_f32.spv"
if "AMEVA_DISABLE_BACKEND" in env:
    del env["AMEVA_DISABLE_BACKEND"]

def parse_output(proc):
    lines = proc.stdout.splitlines()
    transcripts = []
    gpu_disp = 0
    cpu_mulmat = 0
    enc_nodes = 0
    cross_nodes = 0
    total_nodes = 0
    for l in lines:
        if l.startswith("[") and "-->" in l:
            transcripts.append(l)
        # a malformed counter line raises ValueError with the offending text instead of leaving the default
        if "AMEVA_DISPATCH_SUCCESS_COUNT=" in l:
            gpu_disp = int(l.split("=", 1)[1].strip())
        if "CPU_EXECUTED_MUL_MAT_COUNT=" in l:
            cpu_mulmat = int(l.split("=", 1)[1].strip())
        if "AMEVA_SCHEDULED_NODE_COUNT=" in l:
            total_nodes = int(l.split("=", 1)[1].strip())
        if "RUN_1_ENCODER_DISPATCH_COUNT=" in l:
            enc_nodes = int(l.split("=", 1)[1].strip())
        if "RUN_1_CROSS_DISPATCH_COUNT=" in l:
            cross_nodes = int(l.split("=", 1)[1].strip())
    raw = " ".join(transcripts)
    return {
        "rc": proc.returncode,
        "raw": raw,
        "gpu_disp": gpu_disp,
        "cpu_mulmat": cpu_mulmat,
        "total_nodes": total_nodes,
        "enc_nodes": enc_nodes,
        "cross_nodes": cross_nodes,
        "stdout": proc.stdout,
        "stderr": proc.stderr
    }

print("=== RUNNING SMALL F16 ON ENGLISH JFK ===")
cmd_en = [BIN, "-m", MODEL_SMALL, "-f", AUDIO_EN, "-l", "en", "-t", "4", "--no-fallback"]
p_en = subprocess.run(cmd_en, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
res_en = parse_output(p_en)

clean_en = re.sub(r'\[.*?\]', '', res_en["raw"])
clean_en = re.sub(r'[^a-zA-Z0-9\s]', '', clean_en).lower()
words_en = clean_en.split()
ref_words = REF_EN.split()

# Compute WER
d = [[0] * (len(words_en) + 1) for _ in range(len(ref_words) + 1)]
for i in range(len(ref_words) + 1): d[i][0] = i
for j in range(len(words_en) + 1): d[0][j] = j
for i in range(1, len(ref_words) + 1):
    for j in range(1, len(words_en) + 1):
        if ref_words[i-1] == words_en[j-1]: d[i][j] = d[i-1][j-1]
        else: d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
wer = float(d[len(ref_words)][len(words_en)]) / float(len(ref_words))

print(f"SMALL_ENGLISH_RC={res_en['rc']}")
print(f"SMALL_ENGLISH_TRANSCRIPT={clean_en.strip()}")
print(f"SMALL_ENGLISH_WER={wer:.4f}")
print(f"SMALL_ENGLISH_GPU_DISPATCH={res_en['gpu_disp']}")
print(f"SMALL_ENGLISH_CPU_MULMAT={res_en['cpu_mulmat']}")
print(f"SMALL_ENGLISH_SCHEDULED_NODES={res_en['total_nodes']}")

print("\n=== RUNNING SMALL F16 ON KOREAN FIXTURE K02 ===")
cmd_ko = [BIN, "-m", MODEL_SMALL, "-f", AUDIO_KO, "-l", "ko", "-t", "4", "--no-fallback"]
p_ko = subprocess.run(cmd_ko, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
res_ko = parse_output(p_ko)

clean_ko = unicodedata.normalize("NFC", res_ko["raw"])
clean_ko = re.sub(r'\[.*?\]', '', clean_ko)
clean_ko = re.sub(r'[^가-힣0-9a-zA-Z\s]', '', clean_ko).strip()

print(f"SMALL_KOREAN_RC={res_ko['rc']}")
print(f"SMALL_KOREAN_TRANSCRIPT={clean_ko}")
print(f"SMALL_KOREAN_GPU_DISPATCH={res_ko['gpu_disp']}")
print(f"SMALL_KOREAN_CPU_MULMAT={res_ko['cpu_mulmat']}")
print(f"SMALL_KOREAN_SCHEDULED_NODES={res_ko['total_nodes']}")
