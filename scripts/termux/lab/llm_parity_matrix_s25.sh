#!/data/data/com.termux/files/usr/bin/bash
# S25 type matrix: real F16 / Q4_K_M models plus models requantized from them with libllama's quantizer.
set -u
LAB=$HOME/abi-lab
OUT=$LAB/out-qmm
MD=$HOME/models
LM=$HOME/.cache/termux-llamacpp/models
F16=$MD/qwen2.5-0.5b-instruct-fp16.gguf
L1B=$LM/Llama-3.2-1B-Instruct-Q4_K_M.gguf
SUM=$LAB/results/parity_matrix_s25.txt
: > "$SUM"

q() { # IN OUT TYPE [--pure]
    [ -s "$2" ] && return 0
    if LD_LIBRARY_PATH="$OUT" "$OUT/ameva_quantize" "$1" "$2.part" "$3" ${4:-} --threads 6 > "$2.qlog" 2>&1; then
        mv "$2.part" "$2"
        grep -E "^AMEVA_QUANTIZE|model size|quant size" "$2.qlog" | sed 's/^/  /' >> "$SUM"
    else
        echo "QUANT_FAIL $2: $(grep -v '^WARNING: linker' "$2.qlog" | tail -2 | tr '\n' ' ')" >> "$SUM"
        return 1
    fi
}
run() {
    bash "$LAB/llm_parity_run.sh" "$@" >> "$SUM" 2>&1
    # tensor types actually present in the model, from the loader's own summary
    grep -E "llama_model_loader: - type" "$LAB/results/parity_$2.log" | sort -u | head -8 | sed 's/^/  /' >> "$SUM"
}

for T in BF16 Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 Q2_K Q3_K_M Q5_K_M Q6_K IQ4_NL IQ4_XS; do q "$F16" "$MD/q05-$T.gguf" "$T"; done
for T in Q2_K Q3_K_S Q4_K_S Q5_K_S Q6_K IQ4_XS; do q "$L1B" "$MD/l1b-pure-$T.gguf" "$T" --pure; done

run "$F16" s25_q05_F16
for T in BF16 Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 Q2_K Q3_K_M Q5_K_M Q6_K IQ4_NL IQ4_XS; do
    [ -s "$MD/q05-$T.gguf" ] && run "$MD/q05-$T.gguf" "s25_q05_$T" --no-cpu-decode
done
run "$L1B" s25_l1b_Q4_K_M
for T in Q2_K Q3_K_S Q4_K_S Q5_K_S Q6_K IQ4_XS; do
    [ -s "$MD/l1b-pure-$T.gguf" ] && run "$MD/l1b-pure-$T.gguf" "s25_l1b_pure_$T" --no-cpu-decode
done
run "$LM/gemma-2-2b-it-Q4_K_M.gguf" s25_g2b_Q4_K_M --chunks 1
echo MATRIX_DONE >> "$SUM"
