#!/data/data/com.termux/files/usr/bin/bash
# parity_three.sh TAG_PREFIX
# ameva_llm_parity on the three models of ~/.cache/termux-llamacpp/models (Qwen2.5-0.5B, Llama-3.2-1B and
# Gemma-2-2B Q4_K_M; ctx 256 for the two larger ones) with the build in OUT (default ~/abi-lab/out-g2). Start with
# nohup; summaries in ~/abi-lab/results/parity3_TAG_PREFIX.txt, ending with PARITY3_DONE.
set -u
LAB=$HOME/abi-lab
export OUT=${OUT:-$LAB/out-g2}
P=${1:-run}
D=$HOME/.cache/termux-llamacpp/models
S=$LAB/results/parity3_$P.txt
: > "$S"
cd "$LAB" || exit 1
bash llm_parity_run.sh "$D/qwen2.5-0.5b-instruct-q4_k_m.gguf" "${P}_qwen05" >> "$S" 2>&1
bash llm_parity_run.sh "$D/Llama-3.2-1B-Instruct-Q4_K_M.gguf" "${P}_llama1b" --ctx 256 >> "$S" 2>&1
bash llm_parity_run.sh "$D/gemma-2-2b-it-Q4_K_M.gguf" "${P}_gemma2b" --ctx 256 >> "$S" 2>&1
echo PARITY3_DONE >> "$S"
