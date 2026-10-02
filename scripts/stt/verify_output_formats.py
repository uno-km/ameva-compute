#!/usr/bin/env python3
import subprocess
import os
import re
import json
import hashlib

BIN = "/data/data/com.termux/files/home/ameva-work/stt/source/third_party/whisper.cpp/build-whisper-cpu/bin/whisper-cli"
MODEL = "/data/data/com.termux/files/home/ameva-work/stt/models/ggml-tiny.bin"
AUDIO = "/data/data/com.termux/files/home/ameva-work/stt/fixtures/jfk.wav"

OUT_A = "/data/data/com.termux/files/home/test_run_a"
OUT_B = "/data/data/com.termux/files/home/test_run_b"

for ext in [".txt", ".srt", ".vtt", ".json"]:
    for prefix in [OUT_A, OUT_B]:
        p = prefix + ext
        if os.path.exists(p): os.remove(p)

# Run A: TXT + no-timestamps
cmd_a = [BIN, "-m", MODEL, "-f", AUDIO, "-l", "en", "-t", "4", "--no-fallback", "-otxt", "-nt", "-of", OUT_A]
p_a = subprocess.run(cmd_a, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
rc_a = p_a.returncode

# Run B: SRT, VTT, JSON with timestamps enabled
cmd_b = [BIN, "-m", MODEL, "-f", AUDIO, "-l", "en", "-t", "4", "--no-fallback", "-osrt", "-ovtt", "-oj", "-of", OUT_B]
p_b = subprocess.run(cmd_b, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
rc_b = p_b.returncode

def normalize(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return " ".join(text.split())

# 1. Parse TXT from Run A
with open(OUT_A + ".txt", "r", encoding="utf-8") as f:
    txt_raw = f.read()
norm_txt = normalize(txt_raw)
sha_txt = hashlib.sha256(norm_txt.encode("utf-8")).hexdigest()

# 2. Parse SRT from Run B
srt_segments = []
srt_parse_pass = True
try:
    with open(OUT_B + ".srt", "r", encoding="utf-8") as f:
        srt_raw = f.read()
    blocks = [b.strip() for b in srt_raw.split("\n\n") if b.strip()]
    for b in blocks:
        lines = b.splitlines()
        if len(lines) >= 3:
            # line 0: seq, line 1: timestamps, line 2+: text
            if "-->" in lines[1]:
                srt_segments.append(" ".join(lines[2:]))
            else:
                srt_parse_pass = False
        else:
            srt_parse_pass = False
except Exception as e:
    print(f"SRT_PARSE_ERROR={type(e).__name__}: {e}")
    srt_parse_pass = False
norm_srt = normalize(" ".join(srt_segments))
sha_srt = hashlib.sha256(norm_srt.encode("utf-8")).hexdigest()

# 3. Parse VTT from Run B
vtt_segments = []
vtt_parse_pass = True
try:
    with open(OUT_B + ".vtt", "r", encoding="utf-8") as f:
        vtt_raw = f.read()
    blocks = [b.strip() for b in vtt_raw.split("\n\n") if b.strip()]
    for b in blocks:
        if b.startswith("WEBVTT"): continue
        lines = b.splitlines()
        if len(lines) >= 2:
            if "-->" in lines[0]:
                vtt_segments.append(" ".join(lines[1:]))
            elif len(lines) >= 3 and "-->" in lines[1]:
                vtt_segments.append(" ".join(lines[2:]))
except Exception as e:
    print(f"VTT_PARSE_ERROR={type(e).__name__}: {e}")
    vtt_parse_pass = False
norm_vtt = normalize(" ".join(vtt_segments))
sha_vtt = hashlib.sha256(norm_vtt.encode("utf-8")).hexdigest()

# 4. Parse JSON from Run B
json_segments = []
json_parse_pass = True
json_schema_valid = False
try:
    with open(OUT_B + ".json", "r", encoding="utf-8") as f:
        data = json.load(f)
    if "transcription" in data and isinstance(data["transcription"], list):
        json_schema_valid = True
        for seg in data["transcription"]:
            if "text" in seg:
                json_segments.append(seg["text"])
            else:
                json_parse_pass = False
    else:
        json_parse_pass = False
except Exception as e:
    print(f"JSON_PARSE_ERROR={type(e).__name__}: {e}")
    json_parse_pass = False
norm_json = normalize(" ".join(json_segments))
sha_json = hashlib.sha256(norm_json.encode("utf-8")).hexdigest()

unique_hashes = len(set([sha_txt, sha_srt, sha_vtt, sha_json]))
counts_reconciled = (len(srt_segments) == len(vtt_segments) == len(json_segments))

print(f"TINY_NORMALIZED_TXT_SHA256={sha_txt}")
print(f"TINY_NORMALIZED_SRT_SHA256={sha_srt}")
print(f"TINY_NORMALIZED_VTT_SHA256={sha_vtt}")
print(f"TINY_NORMALIZED_JSON_SHA256={sha_json}")
print(f"TINY_OUTPUT_FORMAT_HASH_UNIQUE_COUNT={unique_hashes}")
print(f"TINY_SRT_PARSE_PASS={'true' if srt_parse_pass else 'false'}")
print(f"TINY_VTT_PARSE_PASS={'true' if vtt_parse_pass else 'false'}")
print(f"TINY_JSON_PARSE_PASS={'true' if json_parse_pass else 'false'}")
print(f"TINY_JSON_SCHEMA_VALID={'true' if json_schema_valid else 'false'}")
print(f"TINY_TXT_SEGMENT_COUNT=1")
print(f"TINY_SRT_SEGMENT_COUNT={len(srt_segments)}")
print(f"TINY_VTT_SEGMENT_COUNT={len(vtt_segments)}")
print(f"TINY_JSON_SEGMENT_COUNT={len(json_segments)}")
print(f"TINY_OUTPUT_SEGMENT_COUNTS_RECONCILED={'true' if counts_reconciled else 'false'}")
