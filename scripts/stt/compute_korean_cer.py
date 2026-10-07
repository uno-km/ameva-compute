#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import hashlib
import unicodedata
import re

AUDIO_PATH = "/data/data/com.termux/files/home/korean_test.wav"

def get_sha256(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest()

with open(AUDIO_PATH, "rb") as f:
    audio_sha = hashlib.sha256(f.read()).hexdigest()

ref_raw = "안녕하세요"
# From task-7681 actual output: "[00:00:00.000 --> 00:00:02.000]   (감사합니다)"
hyp_raw = "(감사합니다)"

def normalize_ko(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[^가-힣0-9a-zA-Z\s]', '', text)
    return text.strip()

ref_norm = normalize_ko(ref_raw)
hyp_norm = normalize_ko(hyp_raw)

ref_sha = get_sha256(ref_norm.encode("utf-8"))
hyp_sha = get_sha256(hyp_norm.encode("utf-8"))

def compute_levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,      # deletion
                           dp[i][j-1] + 1,      # insertion
                           dp[i-1][j-1] + cost) # substitution
    
    # Backtrack to count substitutions, deletions, insertions
    i, j = m, n
    subs, dels, ins = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ins += 1
            j -= 1
        else:
            break

    total_dist = dp[m][n]
    cer = float(total_dist) / float(len(s1)) if len(s1) > 0 else 0.0
    return total_dist, subs, dels, ins, cer

dist, subs, dels, ins, cer = compute_levenshtein(ref_norm, hyp_norm)

print(f"KOREAN_AUDIO_SHA256={audio_sha}")
print(f"KOREAN_REFERENCE_TEXT_SHA256={ref_sha}")
print(f"KOREAN_HYPOTHESIS_TEXT_SHA256={hyp_sha}")
print(f"KOREAN_TRANSCRIPT_TEXT={hyp_norm}")
print(f"KOREAN_NORMALIZATION_POLICY=UNICODE_NFC_STRIP_PUNCTUATION_AND_WHITESPACE")
print(f"KOREAN_SUBSTITUTION_COUNT={subs}")
print(f"KOREAN_DELETION_COUNT={dels}")
print(f"KOREAN_INSERTION_COUNT={ins}")
print(f"KOREAN_CER={cer:.4f}")
# Gate: Audit stated KOREAN_ACCURACY=UNPROVEN / CER not yet passed
print(f"KOREAN_CER_GATE=FAIL_ACCURACY_UNPROVEN")
