#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import subprocess
import os
import re
import json
import hashlib
import unicodedata

BIN = "/data/data/com.termux/files/home/ameva-work/stt/source/third_party/whisper.cpp/build-whisper-cpu/bin/whisper-cli"
MODEL_BASE = "/data/data/com.termux/files/home/ameva-work/stt/models/ggml-base.bin"
MODEL_SMALL = "/data/data/com.termux/files/home/ameva-work/stt/models/ggml-small.bin"
AUDIO_JFK = "/data/data/com.termux/files/home/ameva-work/stt/fixtures/jfk.wav"

FIXTURES = [
    ("K01", "안녕하세요", "/data/data/com.termux/files/home/ameva-work/stt/fixtures/korean/k01_hello.wav"),
    ("K02", "아메바 컴퓨트 음성 인식 검증입니다", "/data/data/com.termux/files/home/ameva-work/stt/fixtures/korean/k02_ameva.wav"),
    ("K03", "오늘은 2026년 9월 13일입니다", "/data/data/com.termux/files/home/ameva-work/stt/fixtures/korean/k03_date.wav"),
    ("K04", "서울에서 부산까지 이동합니다", "/data/data/com.termux/files/home/ameva-work/stt/fixtures/korean/k04_travel.wav"),
    ("K05", "대한민국 헌법 제1조 제1항은 대한민국은 민주공화국이다 라고 규정하고 있으며 제2항은 대한민국의 주권은 국민에게 있고 모든 권력은 국민으로부터 나온다고 명시하여 민주주의 기본 원리를 선언합니다", "/data/data/com.termux/files/home/ameva-work/stt/fixtures/korean/k05_constitution.wav")
]

def normalize_ko(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^가-힣0-9a-zA-Z\s]', '', text)
    return " ".join(text.split()).strip()

def compute_cer(ref, hyp):
    ref = re.sub(r'\s+', '', ref)
    hyp = re.sub(r'\s+', '', hyp)
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

def run_whisper(audio_path, model_path, prefix_out, use_cpu_only=False, lang="ko"):
    json_path = prefix_out + ".json"
    if os.path.exists(json_path):
        os.remove(json_path)

    env = os.environ.copy()
    if use_cpu_only:
        env["AMEVA_DISABLE_BACKEND"] = "1"
    else:
        env["AMEVA_SPIRV_F16_PATH"] = "/data/data/com.termux/files/home/matmul_f16_f32.spv"
        env["AMEVA_SPIRV_Q5_0_PATH"] = "/data/data/com.termux/files/home/matmul_q5_0_f32.spv"
        if "AMEVA_DISABLE_BACKEND" in env:
            del env["AMEVA_DISABLE_BACKEND"]

    cmd = [BIN, "-m", model_path, "-f", audio_path, "-l", lang, "-t", "4", "--no-fallback", "-ojf", "-of", prefix_out]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    # Telemetry extraction - STRICT: None if missing
    backend_registered = None
    backend_selected = None
    gpu_dispatches = None
    cpu_mulmats = None
    enc_dispatches = None
    cross_dispatches = None
    dec_steps = None
    dec_dispatches = None
    sched_nodes = None

    lines = proc.stdout.splitlines()
    for l in lines:
        if l.startswith("AMEVA_BACKEND_REGISTERED="):
            backend_registered = (l.split("=")[1].strip() == "true")
        elif l.startswith("AMEVA_BACKEND_SELECTED="):
            backend_selected = l.split("=")[1].strip()
        elif l.startswith("AMEVA_DISPATCH_SUCCESS_COUNT="):
            gpu_dispatches = int(l.split("=")[1].strip())
        elif l.startswith("CPU_EXECUTED_MUL_MAT_COUNT="):
            cpu_mulmats = int(l.split("=")[1].strip())
        elif l.startswith("RUNTIME_ENCODER_DISPATCH_COUNT="):
            enc_dispatches = int(l.split("=")[1].strip())
        elif l.startswith("RUNTIME_CROSS_ATTENTION_DISPATCH_COUNT="):
            cross_dispatches = int(l.split("=")[1].strip())
        elif l.startswith("RUNTIME_DECODER_STEP_COUNT="):
            dec_steps = int(l.split("=")[1].strip())
        elif l.startswith("RUNTIME_DECODER_DISPATCH_COUNT="):
            dec_dispatches = int(l.split("=")[1].strip())
        elif l.startswith("AMEVA_SCHEDULED_NODE_COUNT="):
            sched_nodes = int(l.split("=")[1].strip())

    # Verify telemetry keys present
    telemetry_present = (backend_registered is not None and backend_selected is not None and
                         gpu_dispatches is not None and cpu_mulmats is not None)

    # Extract tokens from JSON
    tokens = []
    transcripts = []
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            jdata = json.load(f)
            for seg in jdata.get("transcription", []):
                transcripts.append(seg.get("text", ""))
                for t in seg.get("tokens", []):
                    tokens.append(t.get("id"))

    full_text = " ".join(transcripts)
    clean_text = normalize_ko(full_text)

    # Token array hash
    token_str = json.dumps(tokens)
    token_sha = hashlib.sha256(token_str.encode("utf-8")).hexdigest()

    return {
        "rc": proc.returncode,
        "clean_text": clean_text,
        "tokens": tokens,
        "token_count": len(tokens),
        "token_sha256": token_sha,
        "backend_registered": backend_registered,
        "backend_selected": backend_selected,
        "gpu_dispatches": gpu_dispatches,
        "cpu_mulmats": cpu_mulmats,
        "enc_dispatches": enc_dispatches,
        "cross_dispatches": cross_dispatches,
        "dec_steps": dec_steps,
        "dec_dispatches": dec_dispatches,
        "sched_nodes": sched_nodes,
        "telemetry_present": telemetry_present
    }

print("=== 1. EVALUATING K01 ~ K05 TOKEN PARITY (BASE F16) ===")
all_cpu_telemetry_ok = True
all_gpu_telemetry_ok = True
korean_results = {}
match_tokens = 0
regressions = 0

for fid, ref_text, audio_path in FIXTURES:
    out_cpu = f"/data/data/com.termux/files/home/korean_{fid}_cpu"
    out_gpu = f"/data/data/com.termux/files/home/korean_{fid}_gpu"

    res_cpu = run_whisper(audio_path, MODEL_BASE, out_cpu, use_cpu_only=True)
    res_gpu = run_whisper(audio_path, MODEL_BASE, out_gpu, use_cpu_only=False)

    if not (res_cpu["telemetry_present"] and res_cpu["backend_registered"] == False and res_cpu["gpu_dispatches"] == 0):
        all_cpu_telemetry_ok = False
    if not (res_gpu["telemetry_present"] and res_gpu["backend_selected"] == "AMEVA0" and res_gpu["gpu_dispatches"] > 0):
        all_gpu_telemetry_ok = False

    is_norm_identical = (res_cpu["clean_text"] == res_gpu["clean_text"])
    is_token_identical = (res_cpu["tokens"] == res_gpu["tokens"])

    div_idx = "None"
    if not is_token_identical:
        for i in range(min(len(res_cpu["tokens"]), len(res_gpu["tokens"]))):
            if res_cpu["tokens"][i] != res_gpu["tokens"][i]:
                div_idx = str(i)
                break
        if div_idx == "None":
            div_idx = str(min(len(res_cpu["tokens"]), len(res_gpu["tokens"])))

    if is_token_identical:
        match_tokens += 1
    else:
        regressions += 1

    _, cer_cpu = compute_cer(ref_text, res_cpu["clean_text"])
    _, cer_gpu = compute_cer(ref_text, res_gpu["clean_text"])

    korean_results[fid] = {
        "ref": ref_text,
        "cpu": res_cpu,
        "gpu": res_gpu,
        "norm_identical": is_norm_identical,
        "token_identical": is_token_identical,
        "div_idx": div_idx,
        "cer_cpu": cer_cpu,
        "cer_gpu": cer_gpu
    }
    print(f"[{fid}] Tokens: CPU={res_cpu['token_count']} GPU={res_gpu['token_count']} | Token Identical: {is_token_identical} | Div: {div_idx}")

print("\n=== 2. DISPATCH RECONCILIATION FOR BASE F16 ON JFK ===")
base_jfk = run_whisper(AUDIO_JFK, MODEL_BASE, "/data/data/com.termux/files/home/base_jfk_gpu", use_cpu_only=False, lang="en")
base_enc = base_jfk["enc_dispatches"]
base_cross = base_jfk["cross_dispatches"]
base_steps = base_jfk["dec_steps"]
base_dec = base_jfk["dec_dispatches"]
base_actual = base_jfk["gpu_dispatches"]
base_mulmat_per_step = (base_dec // base_steps) if base_steps and base_dec is not None else "MISSING"  # no fabricated default (was 49)
base_expected = base_enc + base_cross + base_dec
base_reconciled = (base_expected == base_actual)
base_unattributed = base_actual - base_expected

print("\n=== 3. DISPATCH RECONCILIATION FOR SMALL F16 ON JFK ===")
small_jfk = run_whisper(AUDIO_JFK, MODEL_SMALL, "/data/data/com.termux/files/home/small_jfk_gpu", use_cpu_only=False, lang="en")
small_enc = small_jfk["enc_dispatches"]
small_cross = small_jfk["cross_dispatches"]
small_steps = small_jfk["dec_steps"]
small_dec = small_jfk["dec_dispatches"]
small_actual = small_jfk["gpu_dispatches"]
small_mulmat_per_step = (small_dec // small_steps) if small_steps and small_dec is not None else "MISSING"  # no fabricated default (was 96)
small_expected = small_enc + small_cross + small_dec
small_reconciled = (small_expected == small_actual)
small_unattributed = small_actual - small_expected

print("\n=== 4. SMALL F16 K02 ACCURACY AUDIT ===")
small_k02 = run_whisper(FIXTURES[1][2], MODEL_SMALL, "/data/data/com.termux/files/home/small_k02_gpu", use_cpu_only=False, lang="ko")
_, small_k02_cer = compute_cer(FIXTURES[1][1], small_k02["clean_text"])

print("\n================ AUDIT REPORT OUTPUT ================")
# Aggregated from the parsed runs (None = the key was missing from that run's output). These were constants.
def _agg(side, key):
    vals = [korean_results[f][side][key] for f in korean_results]
    return vals, any(v is None for v in vals)

cpu_reg, cpu_reg_missing = _agg("cpu", "backend_registered")
cpu_disp, cpu_disp_missing = _agg("cpu", "gpu_dispatches")
gpu_sel, gpu_sel_missing = _agg("gpu", "backend_selected")
gpu_disp, gpu_disp_missing = _agg("gpu", "gpu_dispatches")
gpu_cpu_mm, gpu_cpu_mm_missing = _agg("gpu", "cpu_mulmats")
print(f"CPU_BASELINE_AMEVA_BACKEND_REGISTERED={'MISSING' if cpu_reg_missing else str(any(cpu_reg)).lower()}")
print(f"CPU_BASELINE_GPU_DISPATCH_COUNT={'MISSING' if cpu_disp_missing else sum(cpu_disp)}")
print(f"CPU_BASELINE_TELEMETRY_KEYS_PRESENT={str(all_cpu_telemetry_ok).lower()}")
print(f"GPU_AMEVA_BACKEND_SELECTED={'MISSING' if gpu_sel_missing else ','.join(sorted(set(gpu_sel)))}")
print(f"GPU_DISPATCH_COUNT_GT_ZERO={'MISSING' if gpu_disp_missing else str(all(d > 0 for d in gpu_disp)).lower()}")
print(f"GPU_CPU_MULMAT_COUNT={'MISSING' if gpu_cpu_mm_missing else sum(gpu_cpu_mm)}")
print(f"GPU_TELEMETRY_KEYS_PRESENT={str(all_gpu_telemetry_ok).lower()}")

for fid in ["K01", "K02", "K03", "K04", "K05"]:
    kr = korean_results[fid]
    print(f"KOREAN_{fid}_NORMALIZED_TRANSCRIPT_IDENTICAL={str(kr['norm_identical']).lower()}")
    print(f"KOREAN_{fid}_CPU_TOKEN_COUNT={kr['cpu']['token_count']}")
    print(f"KOREAN_{fid}_GPU_TOKEN_COUNT={kr['gpu']['token_count']}")
    print(f"KOREAN_{fid}_CPU_TOKEN_IDS_SHA256={kr['cpu']['token_sha256']}")
    print(f"KOREAN_{fid}_GPU_TOKEN_IDS_SHA256={kr['gpu']['token_sha256']}")
    print(f"KOREAN_{fid}_TOKEN_SEQUENCE_IDENTICAL={str(kr['token_identical']).lower()}")
    print(f"KOREAN_{fid}_FIRST_DIVERGENCE_INDEX={kr['div_idx']}")

print(f"KOREAN_TOKEN_SEQUENCE_MATCH_COUNT={match_tokens}")
print(f"KOREAN_TOKEN_LEVEL_GPU_REGRESSION_COUNT={regressions}")
print(f"KOREAN_RECORDED_FIXTURE_COUNT=0")
print(f"KOREAN_SYNTHETIC_FIXTURE_COUNT=5")

print(f"SMALL_K02_REFERENCE_TEXT={FIXTURES[1][1]}")
print(f"SMALL_K02_TRANSCRIPT_TEXT={small_k02['clean_text']}")
print(f"SMALL_K02_CER={small_k02_cer:.4f}")
small_k02_gpu_ok = small_k02["rc"] == 0 and (small_k02["gpu_dispatches"] or 0) > 0
print(f"SMALL_KOREAN_GPU_EXECUTION={'PASS' if small_k02_gpu_ok else 'FAIL'}")
# no CER threshold is defined for this fixture, so no pass/fail verdict is printed (it was a constant FAIL)
print(f"SMALL_KOREAN_ACCURACY=NOT_JUDGED_NO_CER_THRESHOLD")

print(f"BASE_ENCODER_DISPATCH_COUNT={base_enc}")
print(f"BASE_CROSS_ATTENTION_DISPATCH_COUNT={base_cross}")
print(f"BASE_DECODER_STEP_COUNT={base_steps}")
print(f"BASE_DECODER_MULMAT_PER_STEP={base_mulmat_per_step}")
print(f"BASE_DECODER_DISPATCH_COUNT={base_dec}")
print(f"BASE_EXPECTED_PHYSICAL_DISPATCH_COUNT={base_expected}")
print(f"BASE_ACTUAL_PHYSICAL_DISPATCH_COUNT={base_actual}")
print(f"BASE_RUNTIME_DISPATCH_RECONCILED={str(base_reconciled).lower()}")
print(f"BASE_UNATTRIBUTED_DISPATCH_COUNT={base_unattributed}")

print(f"SMALL_AMEVA_SCHEDULED_NODE_COUNT={small_jfk['sched_nodes'] if small_jfk['sched_nodes'] is not None else 'MISSING'}")
print(f"SMALL_ENCODER_DISPATCH_COUNT={small_enc}")
print(f"SMALL_CROSS_ATTENTION_DISPATCH_COUNT={small_cross}")
print(f"SMALL_DECODER_STEP_COUNT={small_steps}")
print(f"SMALL_DECODER_MULMAT_PER_STEP={small_mulmat_per_step}")
print(f"SMALL_DECODER_DISPATCH_COUNT={small_dec}")
print(f"SMALL_EXPECTED_DISPATCH_COUNT={small_expected}")
print(f"SMALL_ACTUAL_DISPATCH_COUNT={small_actual}")
print(f"SMALL_RUNTIME_DISPATCH_RECONCILED={str(small_reconciled).lower()}")
print(f"SMALL_UNATTRIBUTED_DISPATCH_COUNT={small_unattributed}")

print(f"S22_TINY_F16_STATUS=PENDING_DEPLOYMENT")
print(f"S22_BASE_F16_STATUS=PENDING_DEPLOYMENT")
print(f"S22_SMALL_F16_STATUS=PENDING_DEPLOYMENT")
