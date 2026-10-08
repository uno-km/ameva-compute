#!/data/data/com.termux/files/usr/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Speaker diarization of sherpa-onnx next to Whisper on ggml-ameva: the two stages of termux-stt's hybrid engine
# (termux_stt/engine/hybrid_engine.py: whisper-cli for the text, sherpa-onnx for the speaker intervals,
# SpeakerMapper.align to join them by time). Diarization runs in onnxruntime on the CPU and whisper-cli is a process
# of its own, so the question is whether the two give the same results side by side as alone, and whether the
# segments of the GPU run carry the timestamps the mapper needs.
#
#   bash scripts/termux/lab/stt_diar_check.sh [--out DIR] [--model FILE] [--wav FILE] [--seconds N] [--speakers N]
#                                             [--threads N] [--res DIR]
#
# Steps (result lines start with DIAR_CHECK):
#   1. the first N seconds of the audio as a WAV of its own (one encoder window);
#   2. sherpa-onnx-offline-speaker-diarization alone (pyannote segmentation 3.0 + CAM++ from
#      ~/.cache/termux-stt/models/sherpa; nothing is downloaded, a missing model ends the script);
#   3. whisper-cli alone on AMEVA0 with AMEVA_STRICT=1 and JSON output (segment times);
#   4. 2 and 3 started together; their results are compared with those of 2 and 3;
#   5. SpeakerMapper.align of the installed termux_stt on the segments of 4 and the intervals of 4.
# Exit status 0 only when every step passed. GUARD_MB (default 500) ends the run when MemAvailable falls below it.
set -uo pipefail
OUT="$HOME/abi-lab/out-stt" BUNDLE="$HOME/.local/share/ameva/current/stt"
MODEL="$HOME/.cache/termux-stt/models/whisper/ggml-large-v3-turbo.bin"
WAV="$HOME/.cache/termux-stt/samples/jfk_1min.wav"
SECONDS_N=11 SPEAKERS=2 THREADS=4 RES=""
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --wav) WAV="$2"; shift 2 ;;
        --seconds) SECONDS_N="$2"; shift 2 ;;
        --speakers) SPEAKERS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --res) RES="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
die() { echo "DIAR_CHECK FAIL: $*"; exit 1; }
RES="${RES:-$HOME/abi-lab/results/stt_diar_$(date +%m%d_%H%M%S)}"
mkdir -p "$RES"
GUARD_MB=${GUARD_MB:-500}
CLI="$BUNDLE/bin/whisper-cli"
DIAR="$(command -v sherpa-onnx-offline-speaker-diarization || true)"
SH="$HOME/.cache/termux-stt/models/sherpa"
SEG="$SH/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB="$SH/3dspeaker-campplus"
[ -x "$CLI" ] || die "$CLI not found"
[ -n "$DIAR" ] || die "sherpa-onnx-offline-speaker-diarization is not installed"
[ -f "$SEG" ] || die "segmentation model $SEG not found (no download is attempted)"
[ -f "$EMB" ] || die "embedding model $EMB not found (no download is attempted)"
[ -f "$MODEL" ] && [ -f "$WAV" ] || die "whisper model or audio not found"
[ -f "$OUT/libggml-ameva.so" ] && [ -x "$OUT/ggml_list_devices" ] || die "plugin not found in $OUT: run stt_build.sh"
DEV="$(LD_LIBRARY_PATH="$OUT/ggml-rt:$OUT" GGML_BACKEND_PATH="$OUT/libggml-ameva.so" "$OUT/ggml_list_devices" 2>/dev/null |
       sed -n 's/.*(-dev \([0-9]*\)) AMEVA0 .*/\1/p' | sed -n 1p)"
[ -n "$DEV" ] || die "AMEVA0 is not registered"
avail_mb() { awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo; }
echo "conditions: $(date +%F_%T) device=$(getprop ro.product.model) cpus=$(awk '/Cpus_allowed_list/ {print $2}' /proc/self/status)" \
     "threads=$THREADS model=$(basename "$MODEL") seconds=$SECONDS_N speakers=$SPEAKERS mem_available_mb=$(avail_mb) ameva_dev=$DEV"

# 1. the first N seconds
CUT="$RES/cut.wav"
python3 - "$WAV" "$CUT" "$SECONDS_N" <<'PY' || die "cutting the audio"
import sys, wave
src, dst, sec = sys.argv[1], sys.argv[2], float(sys.argv[3])
with wave.open(src, "rb") as r:
    p = r.getparams()
    frames = r.readframes(int(p.framerate * sec))
with wave.open(dst, "wb") as w:
    w.setparams(p)
    w.writeframes(frames)
PY

run_diar() {  # $1 = tag
    local t0 t1
    t0=$(date +%s.%N)
    "$DIAR" --clustering.num-clusters="$SPEAKERS" --segmentation.pyannote-model="$SEG" --embedding.model="$EMB" \
        --segmentation.num-threads="$THREADS" --embedding.num-threads="$THREADS" "$CUT" > "$RES/$1.diar.log" 2>&1
    echo "rc=$?" > "$RES/$1.diar.rc"
    t1=$(date +%s.%N)
    grep -E '^[0-9.]+ -- [0-9.]+ speaker_[0-9]+' "$RES/$1.diar.log" > "$RES/$1.diar.txt"
    awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.1f\n", b - a}' > "$RES/$1.diar.wall"
}
run_whisper() {  # $1 = tag
    env LD_LIBRARY_PATH="$OUT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" GGML_BACKEND_PATH="$OUT/libggml-ameva.so" AMEVA_STRICT=1 \
        "$CLI" -dev "$DEV" -m "$MODEL" -f "$CUT" -t "$THREADS" -l en -oj -of "$RES/$1.whisper" \
        > "$RES/$1.whisper.txt" 2> "$RES/$1.whisper.log" &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do
        if [ "$(avail_mb)" -lt "$GUARD_MB" ]; then
            echo "GUARD_KILL mem_available_mb=$(avail_mb) (< $GUARD_MB)" > "$RES/$1.guard"
            kill -9 $pid
        fi
        sleep 1
    done
    wait $pid
    echo "rc=$?" > "$RES/$1.whisper.rc"
}
report() {  # $1 = tag
    local enc
    enc="$(awk '/whisper_print_timings: +encode time/ {sub(/.*= */, ""); print $1; exit}' "$RES/$1.whisper.log")"
    echo "DIAR_CHECK $1: diarization $(cat "$RES/$1.diar.rc") intervals=$(grep -c . "$RES/$1.diar.txt") wall_s=$(cat "$RES/$1.diar.wall");" \
         "whisper $(cat "$RES/$1.whisper.rc") backend=$(grep -o 'using [A-Za-z0-9]* backend' "$RES/$1.whisper.log" | sed -n 1p | cut -d' ' -f2)" \
         "encode_ms=${enc:-na} segments=$(grep -c '"offsets"' "$RES/$1.whisper.json" 2>/dev/null) $(cat "$RES/$1.guard" 2>/dev/null)"
}

# 2 and 3. each alone
run_diar alone
run_whisper alone
report alone
# 4. together
run_diar both &
dpid=$!
run_whisper both
wait $dpid
report both

rc=0
for t in alone both; do
    grep -q "rc=0" "$RES/$t.diar.rc" && [ -s "$RES/$t.diar.txt" ] || { echo "DIAR_CHECK FAIL: diarization ($t)"; rc=1; }
    grep -q "rc=0" "$RES/$t.whisper.rc" && grep -q "using AMEVA0 backend" "$RES/$t.whisper.log" && [ -s "$RES/$t.whisper.json" ] ||
        { echo "DIAR_CHECK FAIL: whisper on AMEVA0 ($t): $(grep -vE '^\s*$' "$RES/$t.whisper.log" | tail -n 1 | cut -c1-200)"; rc=1; }
done
if [ $rc -eq 0 ]; then
    cmp -s "$RES/alone.diar.txt" "$RES/both.diar.txt" && echo "DIAR_CHECK speaker intervals alone vs together: identical" ||
        { echo "DIAR_CHECK FAIL: speaker intervals differ between the two runs"; rc=1; }
    cmp -s "$RES/alone.whisper.json" "$RES/both.whisper.json" && echo "DIAR_CHECK whisper segments alone vs together: identical" ||
        { echo "DIAR_CHECK FAIL: whisper segments differ between the two runs"; rc=1; }
fi
# 5. the join termux-stt makes
if [ $rc -eq 0 ]; then
    python3 - "$RES/both.whisper.json" "$RES/both.diar.txt" "$SPEAKERS" <<'PY' || { echo "DIAR_CHECK FAIL: SpeakerMapper.align"; rc=1; }
import json, re, sys
from termux_stt.diarization.mapper import SpeakerMapper
from termux_stt.export.result import Segment
segs = [Segment(start=t["offsets"]["from"] / 1000.0, end=t["offsets"]["to"] / 1000.0, text=t["text"].strip())
        for t in json.load(open(sys.argv[1], encoding="utf-8", errors="replace"))["transcription"]]
spk = [(float(a), float(b), int(c)) for a, b, c in
       re.findall(r"^([0-9.]+) -- ([0-9.]+) speaker_([0-9]+)", open(sys.argv[2]).read(), re.M)]
out = SpeakerMapper().align(segs, spk, num_speakers=int(sys.argv[3]))
for i in spk:
    print("DIAR_CHECK interval %.2f-%.2f speaker_%02d" % i)
unlabelled = 0
for s in out:
    unlabelled += s.speaker is None
    print("DIAR_CHECK aligned %.2f-%.2f %s | %s" % (s.start, s.end, s.speaker, s.text[:90]))
if not out or unlabelled:
    sys.exit(1)
PY
fi
[ $rc -eq 0 ] && echo "DIAR_CHECK PASS $RES" || echo "DIAR_CHECK FAILED $RES"
exit $rc
