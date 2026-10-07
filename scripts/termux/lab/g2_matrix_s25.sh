#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# Graph-engine (out-g2) parity on the S25 model set, flash attention on; summary in results/g2_matrix_s25.txt
set -u
LAB=$HOME/abi-lab
LM=$HOME/.cache/termux-llamacpp/models
MD=$HOME/models
SUM=$LAB/results/g2_matrix_s25.txt
: > "$SUM"
run() { bash "$LAB/llm_parity_run.sh" "$@" >> "$SUM" 2>&1; }
run "$MD/qwen2.5-0.5b-instruct-fp16.gguf" g2_qwen05_F16
run "$LM/Llama-3.2-1B-Instruct-Q4_K_M.gguf" g2_l1b_Q4_K_M
run "$LM/gemma-2-2b-it-Q4_K_M.gguf" g2_g2b_Q4_K_M --chunks 1
for T in Q2_K IQ4_NL Q8_0 BF16; do run "$MD/q05-$T.gguf" "g2_q05_$T" --no-cpu-decode; done
echo MATRIX_DONE >> "$SUM"
