#!/data/data/com.termux/files/usr/bin/bash
set -u

LOGFILE="$HOME/s25_stress_bake.log"
MODEL="$HOME/.cache/termux-llamacpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
LIB="$PREFIX/lib/libggml-ameva.so"

echo "==========================================================" | tee -a "$LOGFILE"
echo "  S25 AMEVA Vulkan Overnight Burn-in Test Started        " | tee -a "$LOGFILE"
echo "  Date: $(date)" | tee -a "$LOGFILE"
echo "  Device: Samsung Galaxy S25 (Snapdragon 8 Elite / Adreno 830)" | tee -a "$LOGFILE"
echo "==========================================================" | tee -a "$LOGFILE"

PROMPTS=(
    "Explain quantum computing in simple terms:"
    "Write a quicksort algorithm in C:"
    "The history of artificial intelligence began with"
    "Why is the sky blue? Explain the physics:"
    "In deep learning, backpropagation works by"
    "Translate the following to French: Knowledge is power."
    "Explain the difference between Vulkan and OpenGL:"
    "Write a Python function to compute Fibonacci numbers:"
    "The primary architectural advantage of RISC-V is"
    "Summarize the theory of general relativity:"
)

TOTAL_PASS=0
TOTAL_FAIL=0

for round in $(seq 1 50); do
    IDX=$(( (round - 1) % ${#PROMPTS[@]} ))
    PROMPT="${PROMPTS[$IDX]}"
    
    echo "" | tee -a "$LOGFILE"
    echo "--- [Round $round / 50] $(date +%H:%M:%S) ---" | tee -a "$LOGFILE"
    echo "Prompt: \"$PROMPT\"" | tee -a "$LOGFILE"
    
    MEM_BEFORE=$(free -m | awk '/Mem:/ {print $3}')
    
    OUT=$(GGML_BACKEND_PATH="$LIB" timeout 45 llama-cli \
        -m "$MODEL" \
        -p "$PROMPT" \
        -n 32 \
        --device AMEVA0 \
        -ngl 99 \
        -fit off \
        --no-display-prompt 2>&1)
    
    RC=$?
    MEM_AFTER=$(free -m | awk '/Mem:/ {print $3}')
    
    SPEED=$(echo "$OUT" | grep -o '\[ Prompt:.*Generation:.* \]' | tail -1)
    SAMPLE_TEXT=$(echo "$OUT" | grep -v 'ggml_vulkan\|WARNING\|load_backend\|ggml-ameva\|Loading model\|build\|model\|modalities\|available commands\|>\|^[[:space:]]*$' | tail -3 | tr '\n' ' ')
    
    if [ $RC -eq 0 ] && [ -n "$SPEED" ]; then
        TOTAL_PASS=$((TOTAL_PASS + 1))
        echo "Status: PASS | Speed: $SPEED | RAM: ${MEM_BEFORE}MB -> ${MEM_AFTER}MB" | tee -a "$LOGFILE"
        echo "Output: $SAMPLE_TEXT" | tee -a "$LOGFILE"
    else
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        echo "Status: FAIL (rc=$RC) | RAM: ${MEM_BEFORE}MB -> ${MEM_AFTER}MB" | tee -a "$LOGFILE"
        echo "$OUT" | tail -10 | tee -a "$LOGFILE"
    fi
    
    sleep 2
done

echo "" | tee -a "$LOGFILE"
echo "==========================================================" | tee -a "$LOGFILE"
echo "  Overnight Burn-in Complete: PASS=$TOTAL_PASS FAIL=$TOTAL_FAIL" | tee -a "$LOGFILE"
echo "  Finished: $(date)" | tee -a "$LOGFILE"
echo "==========================================================" | tee -a "$LOGFILE"
