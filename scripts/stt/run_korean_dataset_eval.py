#!/usr/bin/env python3
import subprocess
import os
import re
import json
import hashlib
import unicodedata
import termux_tts as tts

FIXTURE_DIR = "/data/data/com.termux/files/home/ameva-work/stt/fixtures/korean"
os.makedirs(FIXTURE_DIR, exist_ok=True)

BIN = "/data/data/com.termux/files/home/ameva-work/stt/source/third_party/whisper.cpp/build-whisper-cpu/bin/whisper-cli"
MODEL_BASE = "/data/data/com.termux/files/home/ameva-work/stt/models/ggml-base.bin"

FIXTURES = [
    ("K01", "안녕하세요", "k01_hello.wav"),
    ("K02", "아메바 컴퓨트 음성 인식 검증입니다", "k02_ameva.wav"),
    ("K03", "오늘은 2026년 9월 13일입니다", "k03_date.wav"),
    ("K04", "서울에서 부산까지 이동합니다", "k04_travel.wav"),
    ("K05", "대한민국 헌법 제1조 제1항은 대한민국은 민주공화국이다 라고 규정하고 있으며 제2항은 대한민국의 주권은 국민에게 있고 모든 권력은 국민으로부터 나온다고 명시하여 민주주의 기본 원리를 선언합니다", "k05_constitution.wav")
]

print("=== STEP 1: GENERATING KOREAN AUDIO FIXTURES ===")
engine = tts.load(language="ko")
fixture_meta = []

for fid, ref_text, filename in FIXTURES:
    filepath = os.path.join(FIXTURE_DIR, filename)
    # Synthesize
    res = engine.synthesize(ref_text)
    with open(filepath, "wb") as f:
        f.write(res.wav_bytes)
    
    sha256 = hashlib.sha256(res.wav_bytes).hexdigest()
    ref_norm = unicodedata.normalize("NFC", ref_text)
    ref_norm_clean = re.sub(r'[^가-힣0-9a-zA-Z\s]', '', ref_norm).strip()
    ref_sha = hashlib.sha256(ref_norm_clean.encode('utf-8')).hexdigest()
    
    meta = {
        "id": fid,
        "ref_text": ref_text,
        "ref_clean": ref_norm_clean,
        "ref_sha256": ref_sha,
        "filepath": filepath,
        "bytes": len(res.wav_bytes),
        "sha256": sha256,
        "duration_sec": res.duration_sec,
        "sample_rate": res.sample_rate
    }
    fixture_meta.append(meta)
    print(f"{fid}: {filename} bytes={len(res.wav_bytes)} duration={res.duration_sec:.2f}s sha256={sha256}")

def normalize_ko(text):
    text = unicodedata.normalize("NFC", text)
    # Remove whisper timestamps and special tags e.g. [00:00:00.000 --> ...]
    text = re.sub(r'\[.*?\]', '', text)
    # Strip non-alphanumeric/korean characters except spaces
    text = re.sub(r'[^가-힣0-9a-zA-Z\s]', '', text)
    return " ".join(text.split()).strip()

def compute_cer(ref, hyp):
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    dist = dp[m][n]
    cer = float(dist) / float(len(ref)) if len(ref) > 0 else 0.0
    return dist, cer

def run_whisper(audio_path, model_path, use_cpu_only=False):
    env = os.environ.copy()
    if use_cpu_only:
        env["AMEVA_DISABLE_BACKEND"] = "1"
    else:
        env["AMEVA_SPIRV_F16_PATH"] = "/data/data/com.termux/files/home/matmul_f16_f32.spv"
        env["AMEVA_SPIRV_Q5_0_PATH"] = "/data/data/com.termux/files/home/matmul_q5_0_f32.spv"
        if "AMEVA_DISABLE_BACKEND" in env:
            del env["AMEVA_DISABLE_BACKEND"]

    cmd = [BIN, "-m", model_path, "-f", audio_path, "-l", "ko", "-t", "4", "--no-fallback"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    
    # Parse transcript
    raw_lines = proc.stdout.splitlines()
    transcript_lines = []
    gpu_dispatches = 0
    cpu_mulmats = 0
    for line in raw_lines:
        if line.startswith("[") and "-->" in line:
            transcript_lines.append(line)
        if "AMEVA_DISPATCH_SUCCESS_COUNT=" in line:
            try:
                gpu_dispatches = int(line.split("=")[1])
            except: pass
        if "CPU_EXECUTED_MUL_MAT_COUNT=" in line:
            try:
                cpu_mulmats = int(line.split("=")[1])
            except: pass

    full_transcript = " ".join(transcript_lines)
    clean_text = normalize_ko(full_transcript)
    return {
        "returncode": proc.returncode,
        "raw_lines": transcript_lines,
        "clean_text": clean_text,
        "gpu_dispatches": gpu_dispatches,
        "cpu_mulmats": cpu_mulmats,
        "stdout": proc.stdout,
        "stderr": proc.stderr
    }

print("\n=== STEP 2: RUNNING CPU AND GPU EVALUATION ON BASE F16 ===")
results = []
match_count = 0
regression_count = 0

for meta in fixture_meta:
    fid = meta["id"]
    audio_path = meta["filepath"]
    ref_clean = meta["ref_clean"]
    
    # Run CPU
    res_cpu = run_whisper(audio_path, MODEL_BASE, use_cpu_only=True)
    dist_cpu, cer_cpu = compute_cer(ref_clean, res_cpu["clean_text"])
    
    # Run GPU
    res_gpu = run_whisper(audio_path, MODEL_BASE, use_cpu_only=False)
    dist_gpu, cer_gpu = compute_cer(ref_clean, res_gpu["clean_text"])
    
    is_identical = (res_cpu["clean_text"] == res_gpu["clean_text"])
    if is_identical:
        match_count += 1
    if cer_gpu > cer_cpu:
        regression_count += 1
        
    res_item = {
        "id": fid,
        "ref": ref_clean,
        "cpu_raw": res_cpu["raw_lines"],
        "cpu_clean": res_cpu["clean_text"],
        "cpu_cer": cer_cpu,
        "cpu_mulmats": res_cpu["cpu_mulmats"],
        "gpu_raw": res_gpu["raw_lines"],
        "gpu_clean": res_gpu["clean_text"],
        "gpu_cer": cer_gpu,
        "gpu_dispatches": res_gpu["gpu_dispatches"],
        "identical": is_identical
    }
    results.append(res_item)
    print(f"[{fid}] REF: '{ref_clean}'")
    print(f"      CPU (CER={cer_cpu:.4f}, MulMat={res_cpu['cpu_mulmats']}): '{res_cpu['clean_text']}'")
    print(f"      GPU (CER={cer_gpu:.4f}, Disp={res_gpu['gpu_dispatches']}): '{res_gpu['clean_text']}'")
    print(f"      IDENTICAL: {is_identical}")

mean_cer_cpu = sum(r["cpu_cer"] for r in results) / len(results)
mean_cer_gpu = sum(r["gpu_cer"] for r in results) / len(results)

print("\n=== SUMMARY REPORT ===")
print(f"KOREAN_FIXTURE_COUNT={len(results)}")
for r in results:
    fid = r["id"]
    print(f"KOREAN_{fid}_CPU_TRANSCRIPT={r['cpu_clean']}")
    print(f"KOREAN_{fid}_GPU_TRANSCRIPT={r['gpu_clean']}")
    print(f"KOREAN_{fid}_CPU_CER={r['cpu_cer']:.4f}")
    print(f"KOREAN_{fid}_GPU_CER={r['gpu_cer']:.4f}")
    print(f"KOREAN_{fid}_TOKEN_SEQUENCE_IDENTICAL={str(r['identical']).lower()}")

print(f"KOREAN_CPU_GPU_TRANSCRIPT_MATCH_COUNT={match_count}")
print(f"KOREAN_GPU_REGRESSION_COUNT={regression_count}")
print(f"KOREAN_MEAN_CER_CPU={mean_cer_cpu:.4f}")
print(f"KOREAN_MEAN_CER_GPU={mean_cer_gpu:.4f}")
not_worse = (mean_cer_gpu <= mean_cer_cpu)
print(f"GPU_CER_NOT_WORSE_THAN_CPU={str(not_worse).lower()}")

with open('/data/data/com.termux/files/home/korean_results.json', 'w') as f:
    json.dump({'fixtures': fixture_meta, 'results': results}, f, indent=2, ensure_ascii=False)
