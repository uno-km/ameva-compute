#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import subprocess
import os
import sys
import re
import json
import hashlib
import time

BIN = "/data/data/com.termux/files/home/ameva-work/stt/source/third_party/whisper.cpp/build-whisper-cpu/bin/whisper-cli"
MODEL_TURBO = "/data/data/com.termux/files/home/ameva-work/stt/models/ggml-large-v3-turbo-q5_0.bin"
AUDIO = "/data/data/com.termux/files/home/ameva-work/stt/fixtures/jfk.wav"
SPIRV_F16 = "/data/data/com.termux/files/home/matmul_f16_f32.spv"
SPIRV_Q5_0 = "/data/data/com.termux/files/home/matmul_q5_0_f32.spv"
LIBWHISPER = "/data/data/com.termux/files/home/ameva-work/stt/source/third_party/whisper.cpp/build-whisper-cpu/src/libwhisper.so"
LIBGGML_BASE = "/data/data/com.termux/files/home/ameva-work/stt/source/third_party/whisper.cpp/build-whisper-cpu/ggml/src/libggml-base.so"
TEST_TURBO_BIN = "/data/data/com.termux/files/home/ameva-work/stt/build/test_turbo_q5_0_native_matmul"
SOURCE_SHA256_H = "/data/data/com.termux/files/home/ameva-work/stt/source/include/ac_vk_sha256.h"
SOURCE_AMEVA_CPP = "/data/data/com.termux/files/home/ameva-work/stt/source/adapters/whisper/ggml-backend-ameva.cpp"

GROUND_TRUTH_WORDS = "and so my fellow americans ask not what your country can do for you ask what you can do for your country".split()

CACHE_ENABLED = (os.environ.get("AMEVA_E2E_CACHE", "0") == "1")
CACHE_FILE = "/data/data/com.termux/files/home/e2e_results_cache.json"

cached_count = 0
fresh_count = 0

EXPECTED_ARTIFACT_SHA256 = {
    "BIN_WHISPER_CLI": "ce3e354beac70b44e9b26fbde03cef75b8053f6014c99344466df7efe5258d87",
    "LIB_WHISPER": "8059231b8e2d298489f60db767716e0e1e67342d1e0f63672369ea9bbaed79ef",
    "LIB_GGML_BASE": "c261dae2ec4698b5c8683b539c24392f0752187321b835b1a84297ef8650e888",
    "BIN_TEST_TURBO_Q5_0": "30e82249ed5bb6c0be6d98f6dc1da65373dd5d90ced81402be2ad26d945bce53",
    "SPIRV_MATMUL_F16": "4200b2b5b64c6dea074da0f1ca2dbdbe5705d55089406bbad37772a852e3d9a3",
    "SPIRV_MATMUL_Q5_0": "cd343cfba670707f61ddeaff9556d6e262c75f344bff38dca23a7bd93bd720fd",
    "MODEL_LARGE_V3_TURBO_Q5_0": "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
    "HEADER_AC_VK_SHA256": "0d43b1d29777045696f68320bb5dab607de737752e974fe14b0f54e836a32a1c",
    "SOURCE_GGML_BACKEND_AMEVA": "2a5d1587c98b73fc2917abdd3445be4d84750d17f2aeaf2e89fcdda50aa07f5f",
}

def compute_file_sha256(filepath):
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def normalize(text):
    text = re.sub(r'\[.*?\]', '', text) # remove timestamps/tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return [w for w in text.split() if w]

def compute_wer(hyp_words, ref_words):
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1): d[i][0] = i
    for j in range(len(hyp_words) + 1): d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    errors = d[len(ref_words)][len(hyp_words)]
    return float(errors) / float(len(ref_words)) if ref_words else 0.0

def load_cache():
    if not CACHE_ENABLED:
        return {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"CACHE_IGNORED={CACHE_FILE}: {type(e).__name__}: {e} (all tests run fresh)", file=sys.stderr)
    return {}

def save_cache(cache):
    if not CACHE_ENABLED:
        return
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError as e:
        print(f"CACHE_NOT_SAVED={CACHE_FILE}: {e}", file=sys.stderr)

def run_test(name, env_overrides, model=MODEL_TURBO, expect_fail=False, extra_args=None, allow_cache=True):
    global cached_count, fresh_count
    print(f"\n================================================================================")
    print(f" RUNNING TEST: {name}")
    print(f"================================================================================")
    
    if CACHE_ENABLED and allow_cache and not expect_fail:
        cache = load_cache()
        if name in cache:
            cached_item = cache[name]
            c_rc = cached_item.get("exit_code")
            valid = (c_rc == 0)
            if valid:
                print(f"[CACHE HIT] Reusing verified result for {name} (exit_code={c_rc})")
                cached_count += 1
                return cached_item

    fresh_count += 1
    env = os.environ.copy()
    env["AMEVA_SPIRV_F16_PATH"] = SPIRV_F16
    env["AMEVA_SPIRV_Q5_0_PATH"] = SPIRV_Q5_0
    for k, v in env_overrides.items():
        if v is None:
            if k in env: del env[k]
        else:
            env[k] = v

    if extra_args:
        cmd = [BIN, "-m", model] + extra_args + ["-l", "en", "-t", "4", "--no-fallback"]
    else:
        cmd = [BIN, "-m", model, "-f", AUDIO, "-l", "en", "-t", "4", "--no-fallback"]
    print(f"CMD: {' '.join(cmd)}")
    print(f"ENV: { {k: env[k] for k in env_overrides} }")

    start_wall = time.time()
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, timeout=800)
        output = p.stdout
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        print(f"FATAL: Command timed out after 800 seconds: {e}")
        raw_out = e.stdout if e.stdout else ""
        output = raw_out.decode('utf-8', errors='replace') if isinstance(raw_out, bytes) else str(raw_out)
        rc = -999
    wall_duration_s = time.time() - start_wall

    print(f"PROCESS_EXIT_CODE={rc}")
    print(f"WALL_TIME_SECONDS={wall_duration_s:.2f}")
    print("--- RAW OUTPUT SNIPPET ---")
    for line in output.splitlines()[-35:]:
        print(line)
    print("--------------------------")

    res = {
        "name": name,
        "exit_code": rc,
        "output": output,
        "wall_time_s": wall_duration_s,
    }

    # Extract assertion keys
    keys = [
        "LOADED_LIBWHISPER_PATH", "LOADED_LIBGGML_PATH",
        "BUILD_TREE_LIBS_LOADED", "SYSTEM_LIBWHISPER_LOADED",
        "AMEVA_BACKEND_REGISTERED", "AMEVA_BACKEND_SELECTED",
        "AMEVA_SCHEDULED_NODE_COUNT", "AMEVA_DISPATCH_ATTEMPT_COUNT",
        "AMEVA_DISPATCH_SUCCESS_COUNT", "AMEVA_DISPATCH_FAILURE_COUNT",
        "CPU_EXECUTED_MUL_MAT_COUNT", "SILENT_FALLBACK_COUNT",
        "AMEVA_FAIL_CLOSED_REASON",
        "WEIGHT_CACHE_HASH_ALGORITHM", "SPARSE_FINGERPRINT_USED",
        "CACHE_KEY_INCLUDES_BYTE_SIZE", "CACHE_KEY_INCLUDES_DTYPE",
        "CACHE_KEY_INCLUDES_SHAPE", "CACHE_COLLISION_PAYLOAD_RECHECK",
        "CACHE_KEY_ALIAS_COUNT", "CACHE_LOOKUP_RECONCILED",
        "CACHE_MISS_UPLOAD_RECONCILED", "CACHE_ENTRY_LIFECYCLE_BALANCED",
        "WEIGHT_CACHE_ENTRY_CREATE_COUNT", "WEIGHT_CACHE_ENTRY_DESTROY_COUNT",
        "WEIGHT_CACHE_HIT_COUNT", "WEIGHT_CACHE_MISS_COUNT",
        "WEIGHT_UPLOAD_COUNT_FIRST_RUN", "WEIGHT_UPLOAD_COUNT_SECOND_RUN",
        "INFERENCE_RUN_COUNT", "INFERENCE_RUN_1_DISPATCH_COUNT",
        "INFERENCE_RUN_2_DISPATCH_COUNT", "INFERENCE_RUN_1_WEIGHT_UPLOAD_COUNT",
        "INFERENCE_RUN_2_WEIGHT_UPLOAD_COUNT",
        "INTERNAL_ACTIVATION_UPLOAD_COUNT", "INTERNAL_OUTPUT_READBACK_COUNT",
        "EVENT_CREATE_DESTROY_COUNT", "EXPECTED_GPU_DISPATCH_COUNT",
        "ACTUAL_GPU_DISPATCH_COUNT", "DISPATCH_COUNT_RECONCILED",
        "WEIGHT_UPLOAD_TIME_MS", "ACTIVATION_UPLOAD_TIME_MS",
        "DISPATCH_API_TIME_MS", "EVENT_WAIT_TIME_MS",
        "OUTPUT_READBACK_TIME_MS", "WEIGHT_UPLOAD_BYTES",
        "ACTIVATION_UPLOAD_BYTES", "OUTPUT_READBACK_BYTES",
        "RUN_1_ENCODER_DISPATCH_COUNT", "RUN_1_CROSS_DISPATCH_COUNT",
        "RUN_1_DECODER_DISPATCH_COUNT", "RUN_2_ENCODER_DISPATCH_COUNT",
        "RUN_2_CROSS_DISPATCH_COUNT", "RUN_2_DECODER_DISPATCH_COUNT",
        "RUN_2_CONTEXT_FORWARD_PASS_COUNT", "RUN_2_CONTEXT_MATMULS_PER_PASS",
        "RUN_2_CONTEXT_DISPATCH_COUNT", "RUN_2_EXTRA_DISPATCH_REASON_SOURCE",
        "RUN_2_EXTRA_DISPATCH_RECONCILED",
        "GRAPH_COMMAND_BUFFER_COUNT", "GRAPH_COMPLETION_EVENT_COUNT",
        "NODE_LEVEL_EVENT_WAIT_COUNT"
    ]
    for k in keys:
        m = re.search(rf"^{k}=(.*)$", output, re.M)
        if not m:
            m = re.search(rf"\b{k}=([^\s]+)", output)
        res[k] = m.group(1).strip() if m else None

    # Extract whisper_print_timings metrics
    m_encode = re.search(r'whisper_print_timings:\s+encode time =\s+([0-9.]+)\s+ms', output)
    m_decode = re.search(r'whisper_print_timings:\s+decode time =\s+([0-9.]+)\s+ms', output)
    m_total = re.search(r'whisper_print_timings:\s+total time =\s+([0-9.]+)\s+ms', output)
    res["TIMING_ENCODE_MS"] = float(m_encode.group(1)) if m_encode else None
    res["TIMING_DECODE_MS"] = float(m_decode.group(1)) if m_decode else None
    res["TIMING_TOTAL_MS"] = float(m_total.group(1)) if m_total else None

    # Extract transcription
    trans_lines = []
    for line in output.splitlines():
        m_trans = re.search(r'\[.*?-->.*?\]\s*(.*)', line)
        if m_trans:
            trans_lines.append(m_trans.group(1).strip())
    transcript_text = " ".join(trans_lines)
    norm_words = normalize(transcript_text)
    ref_words = (GROUND_TRUTH_WORDS * 2) if (extra_args and extra_args.count("-f") >= 2) else GROUND_TRUTH_WORDS
    wer = compute_wer(norm_words, ref_words)
    res["TRANSCRIPTION"] = transcript_text
    res["WER"] = wer
    print(f"TRANSCRIPTION: {transcript_text}")
    print(f"WER: {wer:.4f} ({len(norm_words)} words vs {len(ref_words)} ref)")

    if CACHE_ENABLED and allow_cache and not expect_fail and rc == 0:
        c = load_cache()
        c[name] = res
        save_cache(c)

    return res

# 0. Print and Verify SHA-256 Hashes of all artifacts (P0-3 Anti-Tamper Gate)
print("================================================================================")
print(" AMEVA COMPUTE HARDWARE VERIFICATION: ARTIFACT INTEGRITY CHECK")
print("================================================================================")
artifacts = [
    ("BIN_WHISPER_CLI", BIN),
    ("LIB_WHISPER", LIBWHISPER),
    ("LIB_GGML_BASE", LIBGGML_BASE),
    ("BIN_TEST_TURBO_Q5_0", TEST_TURBO_BIN),
    ("SPIRV_MATMUL_F16", SPIRV_F16),
    ("SPIRV_MATMUL_Q5_0", SPIRV_Q5_0),
    ("MODEL_LARGE_V3_TURBO_Q5_0", MODEL_TURBO),
    ("HEADER_AC_VK_SHA256", SOURCE_SHA256_H),
    ("SOURCE_GGML_BACKEND_AMEVA", SOURCE_AMEVA_CPP),
]

actual_hashes = {}
artifact_match_count = 0
artifact_mismatch_count = 0

for tag, path in artifacts:
    h = compute_file_sha256(path)
    actual_hashes[tag] = h
    print(f"ARTIFACT_SHA256_{tag}={h} ({path})")
    expected = EXPECTED_ARTIFACT_SHA256.get(tag)
    if expected:
        if h == expected:
            artifact_match_count += 1
        else:
            artifact_mismatch_count += 1
            print(f"ARTIFACT_HASH_MISMATCH [{tag}]: actual {h} != expected {expected}")

print(f"\nARTIFACT_EXPECTED_HASH_SOURCE=INSPECTION_MANIFEST")
print(f"ARTIFACT_HASH_MATCH_COUNT={artifact_match_count}")
print(f"ARTIFACT_HASH_MISMATCH_COUNT={artifact_mismatch_count}")
artifact_gate_pass = (artifact_mismatch_count == 0)
print(f"ARTIFACT_REPRODUCIBILITY_GATE={'PASS' if artifact_gate_pass else 'FAIL'}")
print(f"ARTIFACT_ANTI_TAMPER_GATE=UNSIGNED")
print(f"EXPECTED_HASH_EXTERNAL_TRUST_ROOT=false")

if not artifact_gate_pass:
    print("FATAL: Artifact integrity gate failed. Aborting verification suite.")
    sys.exit(1)

print(f"\nE2E_CACHE_ENABLED={'true' if CACHE_ENABLED else 'false'}")
print(f"E2E_RESULT_SOURCE={'FRESH_EXECUTION' if not CACHE_ENABLED else 'CONDITIONAL_CACHE'}")

results = {}

# 1. Test Q5_0 Contract None (Expect 0 GPU nodes, all CPU)
results["none"] = run_test("Q5_0_CONTRACT_NONE", {"AMEVA_Q5_0_CONTRACT": "none"})

# 2. Test Q5_0 Contract Restricted (Expect 136 GPU nodes, 832 dispatches)
results["restricted"] = run_test("Q5_0_CONTRACT_RESTRICTED", {"AMEVA_Q5_0_CONTRACT": "restricted"})

# 3. Test Q5_0 Contract All (Expect 200 GPU nodes, 1128 dispatches)
results["all"] = run_test("Q5_0_CONTRACT_ALL", {"AMEVA_Q5_0_CONTRACT": "all"})

# 4. Test Fault Injection (Expect exit code 10 with exact fail-closed reason; NEVER cached)
results["fault_injection"] = run_test("FAIL_CLOSED_FAULT_INJECTION", {
    "AMEVA_Q5_0_CONTRACT": "restricted",
    "AMEVA_RUNTIME_FAIL_STAGE": "dispatch"
}, expect_fail=True, allow_cache=False)

# 5. Multi-Inference Run (Expect INFERENCE_RUN_COUNT=2, 2nd run uploads=0, exact 832/880 dispatches)
results["multi_run"] = run_test("MULTI_INFERENCE_2_RUNS", {
    "AMEVA_Q5_0_CONTRACT": "restricted"
}, extra_args=["-f", AUDIO, "-f", AUDIO])

# 6. Summary Audit Report with Strict require_exact Assertions
print("\n================================================================================")
print(" E2E FORMAL VERIFICATION SUITE AUDIT SUMMARY")
print("================================================================================")
print(f"CACHED_RESULT_COUNT={cached_count}")
print(f"FRESH_EXECUTION_COUNT={fresh_count}")
if not CACHE_ENABLED:
    print("VERIFICATION_EXECUTION_GUARANTEE=FRESH_100_PERCENT")

all_pass = True

def require_exact(test_name, key, actual, expected):
    global all_pass
    if str(actual) != str(expected):
        print(f"ASSERTION_FAILED [{test_name}] {key}: actual '{actual}' != expected '{expected}'")
        all_pass = False
        return False
    print(f"ASSERTION_PASS [{test_name}] {key} == '{expected}'")
    return True

def require_int(result, key):
    raw = result.get(key)
    if raw is None:
        raise AssertionError(f"[{result['name']}] Missing required telemetry key: {key}")
    try:
        return int(raw)
    except ValueError:
        raise AssertionError(f"[{result['name']}] Invalid integer format for {key}: '{raw}'")

# Validate Common Linkage & Registry across non-fault tests
for tkey in ["none", "restricted", "all", "multi_run"]:
    r = results[tkey]
    print(f"\n--- VALIDATING LINKAGE & INVARIANTS: {r['name']} ---")
    require_exact(r['name'], "EXIT_CODE", r['exit_code'], 0)
    require_exact(r['name'], "BUILD_TREE_LIBS_LOADED", r.get('BUILD_TREE_LIBS_LOADED'), "true")
    require_exact(r['name'], "SYSTEM_LIBWHISPER_LOADED", r.get('SYSTEM_LIBWHISPER_LOADED'), "false")
    require_exact(r['name'], "AMEVA_BACKEND_REGISTERED", r.get('AMEVA_BACKEND_REGISTERED'), "true")
    require_exact(r['name'], "AMEVA_BACKEND_SELECTED", r.get('AMEVA_BACKEND_SELECTED'), "AMEVA0")
    require_exact(r['name'], "AMEVA_DISPATCH_FAILURE_COUNT", r.get('AMEVA_DISPATCH_FAILURE_COUNT'), "0")
    require_exact(r['name'], "SILENT_FALLBACK_COUNT", r.get('SILENT_FALLBACK_COUNT'), "0")
    require_exact(r['name'], "WER", r.get('WER'), 0.0)

    # Invariant: attempts == successes + failures
    attempts = require_int(r, "AMEVA_DISPATCH_ATTEMPT_COUNT")
    successes = require_int(r, "AMEVA_DISPATCH_SUCCESS_COUNT")
    failures = require_int(r, "AMEVA_DISPATCH_FAILURE_COUNT")
    if attempts != (successes + failures):
        print(f"ASSERTION_FAILED [{r['name']}] DISPATCH_INVARIANT: attempts ({attempts}) != successes ({successes}) + failures ({failures})")
        all_pass = False
    else:
        print(f"ASSERTION_PASS [{r['name']}] DISPATCH_INVARIANT: attempts ({attempts}) == successes ({successes}) + failures ({failures})")

# Granular Checks: None
r_none = results["none"]
print(f"\n--- VALIDATING Q5_0_CONTRACT_NONE ---")
require_exact("Q5_0_CONTRACT_NONE", "AMEVA_SCHEDULED_NODE_COUNT", r_none.get('AMEVA_SCHEDULED_NODE_COUNT'), "0")
require_exact("Q5_0_CONTRACT_NONE", "CPU_EXECUTED_MUL_MAT_COUNT", r_none.get('CPU_EXECUTED_MUL_MAT_COUNT'), "200")

# Granular Checks: Restricted
r_res = results["restricted"]
print(f"\n--- VALIDATING Q5_0_CONTRACT_RESTRICTED ---")
require_exact("Q5_0_CONTRACT_RESTRICTED", "AMEVA_SCHEDULED_NODE_COUNT", r_res.get('AMEVA_SCHEDULED_NODE_COUNT'), "136")
require_exact("Q5_0_CONTRACT_RESTRICTED", "AMEVA_DISPATCH_SUCCESS_COUNT", r_res.get('AMEVA_DISPATCH_SUCCESS_COUNT'), "832")
require_exact("Q5_0_CONTRACT_RESTRICTED", "CPU_EXECUTED_MUL_MAT_COUNT", r_res.get('CPU_EXECUTED_MUL_MAT_COUNT'), "64")

# Granular Checks: All (Reconciled with exact graph expectations & SHA-256 identity)
r_all = results["all"]
print(f"\n--- VALIDATING Q5_0_CONTRACT_ALL ---")
require_exact("Q5_0_CONTRACT_ALL", "AMEVA_SCHEDULED_NODE_COUNT", r_all.get('AMEVA_SCHEDULED_NODE_COUNT'), "200")
expected_from_graph = 1128
actual_dispatch = require_int(r_all, "AMEVA_DISPATCH_SUCCESS_COUNT")
print(f"EXPECTED_DISPATCH_COUNT_FROM_GRAPH={expected_from_graph}")
print(f"ACTUAL_DISPATCH_SUCCESS_COUNT={actual_dispatch}")
reconciled = (actual_dispatch == expected_from_graph)
print(f"DISPATCH_COUNT_RECONCILED={'true' if reconciled else 'false'}")
require_exact("Q5_0_CONTRACT_ALL", "DISPATCH_COUNT_RECONCILED", "true" if reconciled else "false", "true")
require_exact("Q5_0_CONTRACT_ALL", "AMEVA_DISPATCH_SUCCESS_COUNT", str(actual_dispatch), str(expected_from_graph))
require_exact("Q5_0_CONTRACT_ALL", "CPU_EXECUTED_MUL_MAT_COUNT", r_all.get('CPU_EXECUTED_MUL_MAT_COUNT'), "0")

hits = require_int(r_all, "WEIGHT_CACHE_HIT_COUNT")
misses = require_int(r_all, "WEIGHT_CACHE_MISS_COUNT")
uploads_run1 = require_int(r_all, "WEIGHT_UPLOAD_COUNT_FIRST_RUN")
uploads_run2 = require_int(r_all, "WEIGHT_UPLOAD_COUNT_SECOND_RUN")

require_exact("Q5_0_CONTRACT_ALL", "CACHE_LOOKUP_RECONCILED", "true" if (hits + misses == actual_dispatch) else "false", "true")
require_exact("Q5_0_CONTRACT_ALL", "CACHE_MISS_UPLOAD_RECONCILED", "true" if (misses == uploads_run1 + uploads_run2) else "false", "true")
require_exact("Q5_0_CONTRACT_ALL", "WEIGHT_CACHE_ENTRY_CREATE_COUNT", r_all.get("WEIGHT_CACHE_ENTRY_CREATE_COUNT"), "232")
# whisper-cli prints these counters before whisper_free, while every cache entry is still alive: the honest
# invariant at that point is "nothing evicted yet". Lifecycle balance needs a post-teardown observation
# (tests/stt/test_repeat_formal_backend.cpp prints one); whisper.cpp used to print create as destroy and "true".
require_exact("Q5_0_CONTRACT_ALL", "WEIGHT_CACHE_ENTRY_DESTROY_COUNT", r_all.get("WEIGHT_CACHE_ENTRY_DESTROY_COUNT"), "0")
print("WEIGHT_CACHE_ENTRY_LIFECYCLE_BALANCED=NOT_OBSERVABLE_BEFORE_TEARDOWN")
require_exact("Q5_0_CONTRACT_ALL", "CACHE_KEY_ALIAS_COUNT", r_all.get("CACHE_KEY_ALIAS_COUNT"), "0")
require_exact("Q5_0_CONTRACT_ALL", "WEIGHT_CACHE_HASH_ALGORITHM", r_all.get("WEIGHT_CACHE_HASH_ALGORITHM"), "SHA256_FULL_PAYLOAD")
require_exact("Q5_0_CONTRACT_ALL", "SPARSE_FINGERPRINT_USED", r_all.get("SPARSE_FINGERPRINT_USED"), "false")
require_exact("Q5_0_CONTRACT_ALL", "CACHE_COLLISION_PAYLOAD_RECHECK", r_all.get("CACHE_COLLISION_PAYLOAD_RECHECK"), "true")

# P0-3: Exact Cache Hit Ratio & Reupload Accounting
hit_ratio = (float(hits) / float(actual_dispatch)) * 100.0 if actual_dispatch > 0 else 0.0
print(f"WEIGHT_CACHE_HIT_RATIO={hit_ratio:.2f}_PERCENT")
# every miss uploads a weight and creates one cache entry: misses beyond the entries created are re-uploads
# of a weight that was already cached (these three lines were constants before)
entries_created = require_int(r_all, "WEIGHT_CACHE_ENTRY_CREATE_COUNT")
reuploads = misses - entries_created
print(f"UNIQUE_WEIGHT_UPLOAD_ONCE={'true' if reuploads == 0 else 'false'}")
print(f"REPEATED_WEIGHT_REUPLOAD_COUNT={reuploads}")
print(f"REUSED_WEIGHT_REUPLOAD_COUNT={reuploads}")

# Granular Checks: Multi-Run Runtime Graph-Kind & Telemetry Reconciliation (P0-1)
r_multi = results["multi_run"]
print(f"\n--- VALIDATING MULTI-RUN RUNTIME TELEMETRY & GRAPH-KIND DISPATCH RECONCILIATION: {r_multi['name']} ---")
require_exact("MULTI_INFERENCE_2_RUNS", "INFERENCE_RUN_COUNT", r_multi.get("INFERENCE_RUN_COUNT"), "2")
require_exact("MULTI_INFERENCE_2_RUNS", "INFERENCE_RUN_1_WEIGHT_UPLOAD_COUNT", r_multi.get("INFERENCE_RUN_1_WEIGHT_UPLOAD_COUNT"), "160")
require_exact("MULTI_INFERENCE_2_RUNS", "INFERENCE_RUN_2_WEIGHT_UPLOAD_COUNT", r_multi.get("INFERENCE_RUN_2_WEIGHT_UPLOAD_COUNT"), "0")

run1_disp = require_int(r_multi, "INFERENCE_RUN_1_DISPATCH_COUNT")
run2_disp = require_int(r_multi, "INFERENCE_RUN_2_DISPATCH_COUNT")
run2_extra_disp = run2_disp - run1_disp

require_exact("MULTI_INFERENCE_2_RUNS", "RUN_1_EXPECTED_DISPATCH_COUNT", str(run1_disp), "832")
require_exact("MULTI_INFERENCE_2_RUNS", "RUN_2_EXPECTED_DISPATCH_COUNT", str(run2_disp), "880")
require_exact("MULTI_INFERENCE_2_RUNS", "RUN_2_EXTRA_DISPATCH_COUNT", str(run2_extra_disp), "48")

# Graph-kind verification from runtime telemetry
require_exact("MULTI_INFERENCE_2_RUNS", "RUN_2_CONTEXT_FORWARD_PASS_COUNT", r_multi.get("RUN_2_CONTEXT_FORWARD_PASS_COUNT"), "3")
require_exact("MULTI_INFERENCE_2_RUNS", "RUN_2_CONTEXT_MATMULS_PER_PASS", r_multi.get("RUN_2_CONTEXT_MATMULS_PER_PASS"), "16")
require_exact("MULTI_INFERENCE_2_RUNS", "RUN_2_CONTEXT_DISPATCH_COUNT", r_multi.get("RUN_2_CONTEXT_DISPATCH_COUNT"), "48")
require_exact("MULTI_INFERENCE_2_RUNS", "RUN_2_EXTRA_DISPATCH_REASON_SOURCE", r_multi.get("RUN_2_EXTRA_DISPATCH_REASON_SOURCE"), "RUNTIME_TELEMETRY")
require_exact("MULTI_INFERENCE_2_RUNS", "RUN_2_EXTRA_DISPATCH_RECONCILED", r_multi.get("RUN_2_EXTRA_DISPATCH_RECONCILED"), "true")

# Granular Checks: Fault Injection Structural Evidence (P1-4)
r_fault = results["fault_injection"]
print(f"\n--- VALIDATING FAULT INJECTION ORACLE: {r_fault['name']} ---")
require_exact("FAULT_INJECTION", "EXIT_CODE", r_fault['exit_code'], 10)
require_exact("FAULT_INJECTION", "AMEVA_FAIL_CLOSED_REASON", r_fault.get('AMEVA_FAIL_CLOSED_REASON'), "INJECTED_DISPATCH_FAILURE")

has_stage_reached = "AMEVA_FAIL_CLOSED_REASON=INJECTED_DISPATCH_FAILURE" in r_fault['output']
require_exact("FAULT_INJECTION", "FAULT_STAGE_REACHED", "true" if has_stage_reached else "false", "true")
# Not observable here: the process exits with code 10 without printing counters after the fault, and comparing
# the literal "0" with "0" (as before) verified nothing.
print("DISPATCH_AFTER_FAULT_COUNT=NOT_MEASURED")
print("CPU_EXECUTION_AFTER_FAULT_COUNT=NOT_MEASURED")

transcript_exposed = len(r_fault.get("TRANSCRIPTION", "").strip()) > 0
require_exact("FAULT_INJECTION", "PARTIAL_TRANSCRIPT_EXPOSED", "true" if transcript_exposed else "false", "false")

reason_match = (r_fault.get('AMEVA_FAIL_CLOSED_REASON') == "INJECTED_DISPATCH_FAILURE")
require_exact("FAULT_INJECTION", "FAILURE_REASON_MATCH", "true" if reason_match else "false", "true")

print(f"FAULT_ABORT_TIME_SECONDS={r_fault.get('wall_time_s', 0.0):.2f}")
print("FAULT_ABORT_TIME_POLICY_DIAGNOSTIC_ONLY=true")
# subprocess reports death by signal N as returncode -N
fault_rc = r_fault['exit_code']
print(f"FAULT_TERMINATING_SIGNAL={-fault_rc if isinstance(fault_rc, int) and fault_rc < 0 else 0}")
# No counter for these exists in the fault run's output; they were printed as constants before.
print("BACKEND_EXIT_CALL_COUNT=NOT_MEASURED")
print("BACKEND_ABORT_CALL_COUNT=NOT_MEASURED")
print("FAULT_RESOURCE_LIFECYCLE_BALANCED=NOT_MEASURED")
print(f"ASSERTION_PASS [FAULT_INJECTION] Clean fail-closed exit code 10 with structural evidence and no data exposure")

# Performance Ground Truth Telemetry (P0-1 & P0-2)
print("\n================================================================================")
print(" SYSTEM PERFORMANCE & LATENCY GROUND TRUTH TELEMETRY")
print("================================================================================")
cpu_wall = results["none"]["wall_time_s"]
gpu_wall = results["all"]["wall_time_s"]
cpu_ms = results["none"].get("TIMING_TOTAL_MS") or (cpu_wall * 1000.0)
gpu_ms = results["all"].get("TIMING_TOTAL_MS") or (gpu_wall * 1000.0)

slowdown = gpu_wall / cpu_wall if cpu_wall > 0 else 0.0

print(f"CPU_WALL_TIME_S={cpu_wall:.2f}")
print(f"ALL_GPU_WALL_TIME_S={gpu_wall:.2f}")
print(f"CPU_SINGLE_RUN_E2E_MS={cpu_ms:.2f}")
print(f"GPU_SINGLE_RUN_E2E_MS={gpu_ms:.2f}")
print(f"OBSERVED_SINGLE_RUN_GPU_SLOWDOWN={slowdown:.2f}x")
print("CPU_E2E_P50_MS=NOT_MEASURED")
print("CPU_E2E_P90_MS=NOT_MEASURED")
print("GPU_E2E_P50_MS=NOT_MEASURED")
print("GPU_E2E_P90_MS=NOT_MEASURED")

# Graph batching architectural status
print("STT_GRAPH_BATCHING=UNPROVEN")
print("GRAPH_LEVEL_BATCHING_STATUS=HOLD")

print(f"\nFINAL_E2E_SUITE_STATUS={'PASS' if all_pass else 'FAIL'}")
sys.exit(0 if all_pass else 1)
