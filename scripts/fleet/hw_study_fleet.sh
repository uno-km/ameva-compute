#!/usr/bin/env bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# hw_study_fleet.sh: run the hardware study of the prompt MUL_MAT (scripts/termux/lab/hw_study_suite.sh) on every
# fleet phone that answers, from the PC (Git Bash), without waiting loops on the phones (each check is one ssh call).
#
#   hw_study_fleet.sh up                       which phones answer ssh
#   hw_study_fleet.sh kit                      make the kit: one backend build pulled from BUILD_DEV:BUILD_DIR
#                                              (default s21:abi-lab/out-tile) plus the study sources of this tree
#                                              (LF), into evidence/hw_study/kits/<id>.tar
#   hw_study_fleet.sh push DEV...              upload the newest kit (checked by sha256) to ~/abi-lab/hwkit/<id>
#   hw_study_fleet.sh run DEV...               take a wake lock and start the suite in the background (nohup);
#                                              STAGES, ROUNDS, SHAPES, LAYOUTS, APIS, TMU_MAX are passed on, and
#                                              MODEL_<dev> (e.g. MODEL_a53=/path/x.gguf) adds the llm stage there
#   hw_study_fleet.sh status DEV...            one line per phone: running / done rc=<n> / no run
#   hw_study_fleet.sh fetch DEV...             copy the newest results to evidence/hw_study/<dev>/<run>/, release
#                                              the wake lock, print the summary (hw_study_summary.py)
#   hw_study_fleet.sh go [DEV...]              kit (if none) + push + run on each phone of DEV (default: the fleet)
#                                              that answers; the phones that do not answer are listed
#
# Rules this follows (fleet memory): scp from the phones hangs, so files move through ssh cat; no sleep loop runs
# inside ssh; the phones suspend unpolled jobs, so run takes the Termux wake lock and fetch releases it.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EV=$ROOT/evidence/hw_study
FLEET=${FLEET:-s21 a35 a53 s20 s25}
BUILD_DEV=${BUILD_DEV:-s21}
BUILD_DIR=${BUILD_DIR:-abi-lab/out-tile}
RT_DIR=${RT_DIR:-abi-lab/rt098}
SSH="ssh -o ConnectTimeout=6 -o BatchMode=yes"
mkdir -p "$EV/kits"

up() { timeout 10 $SSH "$1" true > /dev/null 2>&1; }

newest_kit() { ls -t "$EV"/kits/*.tar 2>/dev/null | head -1; }

cmd_up() { for d in $FLEET; do if up "$d"; then echo "$d up"; else echo "$d down"; fi; done; }

cmd_kit() {
    up "$BUILD_DEV" || { echo "KIT FAIL: build phone $BUILD_DEV does not answer"; return 1; }
    local tmp; tmp=$(mktemp -d)
    mkdir -p "$tmp/kit/bin" "$tmp/kit/rt" "$tmp/kit/src/include"
    # the backend build and its ggml / llama libraries, as they are on the build phone
    $SSH "$BUILD_DEV" "cd ~/$BUILD_DIR && tar -chf - --hard-dereference libggml-ameva.so libac_runtime.so.2 ameva_qmm_check ameva_mmq_bench ggml-rt" \
        | tar -xf - -C "$tmp/kit/bin" || { echo "KIT FAIL: cannot read ~/$BUILD_DIR on $BUILD_DEV"; rm -rf "$tmp"; return 1; }
    $SSH "$BUILD_DEV" "cd ~/$RT_DIR && tar -chf - --hard-dereference ." | tar -xf - -C "$tmp/kit/rt" || { echo "KIT FAIL: no ~/$RT_DIR"; rm -rf "$tmp"; return 1; }
    $SSH "$BUILD_DEV" "head -c 65536 ~/abi-lab/corpus/wiki.test.raw" > "$tmp/kit/src/prompt_corpus.txt"
    local mark; mark=$($SSH "$BUILD_DEV" "strings ~/$BUILD_DIR/libggml-ameva.so | grep -c 'Mali OpenCL default'")
    # the sources of this tree with LF line ends (the Windows working tree may be CRLF)
    local f
    for f in cl_hw_bench.c cpu_bw_bench.c gpu_sysfs.sh hw_study_suite.sh llm_plugin_ab.sh decode_roofline.sh; do
        sed 's/\r$//' "$ROOT/scripts/termux/lab/$f" > "$tmp/kit/src/$f"
    done
    cp -r "$ROOT/include/CL" "$tmp/kit/src/include/"
    find "$tmp/kit/src/include" -type f -exec sed -i 's/\r$//' {} +
    local id; id="kit-$(date +%Y%m%d-%H%M%S)-$(git -C "$ROOT" rev-parse --short HEAD)"
    {
        echo "$id"
        echo "built_on=$BUILD_DEV:$BUILD_DIR mali_default_marker=$mark"
        echo "tree=$(git -C "$ROOT" rev-parse HEAD) dirty_lab_files=$(git -C "$ROOT" status --short -- scripts/termux/lab | wc -l)"
        (cd "$tmp/kit" && find bin rt -type f | sort | xargs sha256sum)
    } > "$tmp/kit/KIT_ID"
    tar -cf "$EV/kits/$id.tar" -C "$tmp" kit
    rm -rf "$tmp"
    echo "KIT $EV/kits/$id.tar sha256=$(sha256sum < "$EV/kits/$id.tar" | cut -c1-16) mali_default_marker=$mark"
}

cmd_push() {
    local k; k=$(newest_kit); [ -n "$k" ] || { echo "PUSH FAIL: no kit (run: $0 kit)"; return 1; }
    local id; id=$(basename "$k" .tar); local sum; sum=$(sha256sum < "$k" | cut -d' ' -f1)
    local d rc=0
    for d in "$@"; do
        up "$d" || { echo "PUSH $d down"; rc=1; continue; }
        $SSH "$d" "mkdir -p ~/abi-lab/hwkit && cat > ~/abi-lab/hwkit/$id.tar" < "$k"
        local got; got=$($SSH "$d" "sha256sum ~/abi-lab/hwkit/$id.tar | cut -d' ' -f1")
        if [ "$got" != "$sum" ]; then echo "PUSH FAIL $d: sha256 $got != $sum"; rc=1; continue; fi
        $SSH "$d" "cd ~/abi-lab/hwkit && rm -rf $id && mkdir $id && tar -xf $id.tar -C $id --strip-components=1 && rm $id.tar && { [ -L current ] || rm -rf current; } && ln -sfn $id current && chmod +x $id/bin/ameva_* && echo ok" \
            | sed "s/^/PUSH $d $id /"
    done
    return $rc
}

cmd_run() {
    local d rc=0
    for d in "$@"; do
        up "$d" || { echo "RUN $d down"; rc=1; continue; }
        local mv="MODEL_$d"; local model=${!mv:-}
        local envs=""
        local v
        for v in STAGES ROUNDS SHAPES LAYOUTS APIS TMU_MAX COOL_MAX REPS LLM_ROUNDS; do
            [ -n "${!v:-}" ] && envs="$envs $v='${!v}'"
        done
        [ -n "$model" ] && envs="$envs MODEL='$model'"
        $SSH "$d" "if [ -e ~/abi-lab/hwkit/RUNNING ] && kill -0 \$(cat ~/abi-lab/hwkit/RUNNING) 2>/dev/null; then echo 'RUN $d busy (a suite is running)'; exit 3; fi
            am startservice -a com.termux.service_wake_lock com.termux/.app.TermuxService > /dev/null 2>&1
            cd ~/abi-lab/hwkit && OUT=\$HOME/abi-lab/hwkit/current/results/\$(date +%Y%m%d-%H%M%S) && mkdir -p \$OUT && echo \$OUT > LAST
            nohup env $envs OUT=\$OUT bash current/src/hw_study_suite.sh current < /dev/null > \$OUT/nohup.log 2>&1 &
            echo \$! > RUNNING; echo \"RUN $d started pid=\$! out=\$OUT\"" || rc=1
    done
    return $rc
}

cmd_status() {
    local d
    for d in "$@"; do
        up "$d" || { echo "STATUS $d down"; continue; }
        $SSH "$d" "cd ~/abi-lab/hwkit 2>/dev/null || { echo 'STATUS $d no kit'; exit 0; }
            o=\$(cat LAST 2>/dev/null); [ -n \"\$o\" ] || { echo 'STATUS $d no run'; exit 0; }
            if [ -f \$o/suite.rc ]; then echo \"STATUS $d done rc=\$(cat \$o/suite.rc) \$o\";
            elif kill -0 \$(cat RUNNING 2>/dev/null) 2>/dev/null; then echo \"STATUS $d running: \$(tail -1 \$o/suite.log 2>/dev/null)\";
            else echo \"STATUS $d ended without suite.rc: \$(tail -2 \$o/nohup.log 2>/dev/null | tr '\n' ' ')\"; fi"
    done
}

cmd_fetch() {
    local d
    for d in "$@"; do
        up "$d" || { echo "FETCH $d down"; continue; }
        local o; o=$($SSH "$d" "cat ~/abi-lab/hwkit/LAST 2>/dev/null")
        [ -n "$o" ] || { echo "FETCH $d: no run"; continue; }
        local dst="$EV/$d/$(basename "$o")"; mkdir -p "$dst"
        $SSH "$d" "cd $o && tar -cf - --exclude=cl_hw_bench --exclude=icd ." | tar -xf - -C "$dst"
        $SSH "$d" "[ -f $o/suite.rc ] && am startservice -a com.termux.service_wake_unlock com.termux/.app.TermuxService > /dev/null 2>&1; true"
        echo "FETCH $d -> $dst ($(ls "$dst" | wc -l) files, suite.rc=$(cat "$dst/suite.rc" 2>/dev/null || echo running))"
        python "$ROOT/scripts/fleet/hw_study_summary.py" "$dst" || true
    done
}

cmd_go() {
    local devs=("$@"); [ ${#devs[@]} -gt 0 ] || read -r -a devs <<< "$FLEET"
    local live=() d
    for d in "${devs[@]}"; do if up "$d"; then live+=("$d"); else echo "GO $d down (not run)"; fi; done
    [ ${#live[@]} -gt 0 ] || { echo "GO: no phone answers"; return 1; }
    [ -n "$(newest_kit)" ] || cmd_kit || return 1
    cmd_push "${live[@]}" && cmd_run "${live[@]}"
}

c=${1:-}; shift || true
case $c in
    up) cmd_up ;;
    kit) cmd_kit ;;
    push) cmd_push "$@" ;;
    run) cmd_run "$@" ;;
    status) cmd_status "$@" ;;
    fetch) cmd_fetch "$@" ;;
    go) cmd_go "$@" ;;
    *) sed -n '4,24p' "$0"; exit 2 ;;
esac
