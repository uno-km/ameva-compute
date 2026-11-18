#!/usr/bin/env bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-AMEVA-Commercial
# regress_fleet.sh: the regression checks of one backend build (scripts/termux/lab/regress_suite.sh: ABI, batch KAT,
# 516 MUL_MAT cases, a real model's greedy text against the CPU with prompt / generation t/s) on every fleet phone
# that answers, from the PC (Git Bash). Each step is one ssh call: no wait loop runs on a phone.
#
#   regress_fleet.sh up                 which phones answer ssh
#   regress_fleet.sh kit                one kit from the build phone: BUILD_DEV:BUILD_DIR (default s21:abi-lab/out-s2,
#                                       the output of build_runtime_backend.sh) + docs/abi + the suite of this tree
#                                       -> evidence/regress/kits/<id>.tar
#   regress_fleet.sh push DEV...        upload the newest kit (sha256 checked) to ~/abi-lab/regkit/<id>
#   regress_fleet.sh run DEV...         wake lock, start the suite with nohup (CHECKS, MODEL_<dev> passed on)
#   regress_fleet.sh status DEV...      running / done rc=<failures> / no run
#   regress_fleet.sh fetch DEV...       results to evidence/regress/<dev>/<run>/, wake lock released, summary printed
#   regress_fleet.sh go [DEV...]        kit (if none) + push + run on every phone of DEV (default: the fleet) that answers
#   regress_fleet.sh report             one table of the newest fetched run of every phone
#
# A phone that does not answer is listed as not run; it is never counted as a pass.
# Rules this follows (fleet memory): files move through ssh cat (scp from the phones hangs); the phones suspend
# unpolled jobs, so run takes the Termux wake lock and fetch releases it; one GPU job per phone at a time.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EV=$ROOT/evidence/regress
FLEET=${FLEET:-s21 a35 a53 s20 s25}
BUILD_DEV=${BUILD_DEV:-s21}
BUILD_DIR=${BUILD_DIR:-abi-lab/out-s2}
SSH="ssh -o ConnectTimeout=6 -o BatchMode=yes"
mkdir -p "$EV/kits"

up() { timeout 10 $SSH "$1" true > /dev/null 2>&1; }
newest_kit() { ls -t "$EV"/kits/*.tar 2> /dev/null | head -1; }

cmd_up() { for d in $FLEET; do if up "$d"; then echo "$d up"; else echo "$d down"; fi; done; }

cmd_kit() {
    up "$BUILD_DEV" || { echo "KIT FAIL: build phone $BUILD_DEV does not answer"; return 1; }
    local tmp; tmp=$(mktemp -d)
    mkdir -p "$tmp/kit/bin" "$tmp/kit/abi" "$tmp/kit/src"
    $SSH "$BUILD_DEV" "cd ~/$BUILD_DIR && tar -chf - --hard-dereference libggml-ameva.so libac_runtime.so.2 ameva_qmm_check \
        ac_abi_check test_batch_graph_kat backend-manifest.json ggml-rt" | tar -xf - -C "$tmp/kit/bin" \
        || { echo "KIT FAIL: cannot read ~/$BUILD_DIR on $BUILD_DEV (needs ac_abi_check: build_runtime_backend.sh of this tree)"; rm -rf "$tmp"; return 1; }
    $SSH "$BUILD_DEV" "head -c 65536 ~/abi-lab/corpus/wiki.test.raw" > "$tmp/kit/src/prompt_corpus.txt"
    sed 's/\r$//' "$ROOT/scripts/termux/lab/regress_suite.sh" > "$tmp/kit/src/regress_suite.sh"
    for f in ac_runtime_layout.txt ac_runtime_symbols.txt; do sed 's/\r$//' "$ROOT/docs/abi/$f" > "$tmp/kit/abi/$f"; done
    local id; id="regkit-$(date +%Y%m%d-%H%M%S)-$(git -C "$ROOT" rev-parse --short HEAD)"
    {
        echo "$id"
        echo "built_on=$BUILD_DEV:$BUILD_DIR tree=$(git -C "$ROOT" rev-parse HEAD) dirty=$(git -C "$ROOT" status --short | wc -l)"
        (cd "$tmp/kit" && find bin -type f | sort | xargs sha256sum)
    } > "$tmp/kit/KIT_ID"
    tar -cf "$EV/kits/$id.tar" -C "$tmp" kit
    rm -rf "$tmp"
    echo "KIT $EV/kits/$id.tar sha256=$(sha256sum < "$EV/kits/$id.tar" | cut -c1-16)"
}

cmd_push() {
    local k; k=$(newest_kit); [ -n "$k" ] || { echo "PUSH FAIL: no kit (run: $0 kit)"; return 1; }
    local id; id=$(basename "$k" .tar); local sum; sum=$(sha256sum < "$k" | cut -d' ' -f1)
    local d rc=0
    for d in "$@"; do
        up "$d" || { echo "PUSH $d down"; rc=1; continue; }
        $SSH "$d" "mkdir -p ~/abi-lab/regkit && cat > ~/abi-lab/regkit/$id.tar" < "$k"
        local got; got=$($SSH "$d" "sha256sum ~/abi-lab/regkit/$id.tar | cut -d' ' -f1")
        if [ "$got" != "$sum" ]; then echo "PUSH FAIL $d: sha256 $got != $sum"; rc=1; continue; fi
        $SSH "$d" "cd ~/abi-lab/regkit && rm -rf $id && mkdir $id && tar -xf $id.tar -C $id --strip-components=1 && rm $id.tar \
            && ln -sfn $id current && chmod +x $id/bin/* && echo ok" | sed "s/^/PUSH $d $id /"
    done
    return $rc
}

cmd_run() {
    local d rc=0
    for d in "$@"; do
        up "$d" || { echo "RUN $d down"; rc=1; continue; }
        local mv="MODEL_$d"; local envs=""
        [ -n "${!mv:-}" ] && envs="$envs MODEL='${!mv}'"
        [ -n "${CHECKS:-}" ] && envs="$envs CHECKS='$CHECKS'"
        $SSH "$d" "if [ -e ~/abi-lab/regkit/RUNNING ] && kill -0 \$(cat ~/abi-lab/regkit/RUNNING) 2>/dev/null; then echo 'RUN $d busy'; exit 3; fi
            am startservice -a com.termux.service_wake_lock com.termux/.app.TermuxService > /dev/null 2>&1
            cd ~/abi-lab/regkit && OUT=\$HOME/abi-lab/regkit/current/results/\$(date +%Y%m%d-%H%M%S) && mkdir -p \$OUT && echo \$OUT > LAST
            nohup env $envs OUT=\$OUT bash current/src/regress_suite.sh current < /dev/null > \$OUT/nohup.log 2>&1 &
            echo \$! > RUNNING; echo \"RUN $d started pid=\$! out=\$OUT\"" || rc=1
    done
    return $rc
}

cmd_status() {
    local d
    for d in "$@"; do
        up "$d" || { echo "STATUS $d down"; continue; }
        $SSH "$d" "cd ~/abi-lab/regkit 2>/dev/null || { echo 'STATUS $d no kit'; exit 0; }
            o=\$(cat LAST 2>/dev/null); [ -n \"\$o\" ] || { echo 'STATUS $d no run'; exit 0; }
            if [ -f \$o/suite.rc ]; then echo \"STATUS $d done failures=\$(cat \$o/suite.rc) \$o\";
            elif kill -0 \$(cat RUNNING 2>/dev/null) 2>/dev/null; then echo \"STATUS $d running: \$(tail -1 \$o/summary.txt 2>/dev/null)\";
            else echo \"STATUS $d ended without suite.rc: \$(tail -2 \$o/nohup.log 2>/dev/null | tr '\n' ' ')\"; fi"
    done
}

cmd_fetch() {
    local d
    for d in "$@"; do
        up "$d" || { echo "FETCH $d down"; continue; }
        local o; o=$($SSH "$d" "cat ~/abi-lab/regkit/LAST 2>/dev/null")
        [ -n "$o" ] || { echo "FETCH $d: no run"; continue; }
        local dst="$EV/$d/$(basename "$o")"; mkdir -p "$dst"
        $SSH "$d" "cd $o && tar -cf - --exclude=pc.bin ." | tar -xf - -C "$dst"
        $SSH "$d" "[ -f $o/suite.rc ] && am startservice -a com.termux.service_wake_unlock com.termux/.app.TermuxService > /dev/null 2>&1; true"
        echo "FETCH $d -> $dst (failures=$(cat "$dst/suite.rc" 2> /dev/null || echo running))"
        grep -E '^(device|CHECK|  run|REGRESS)' "$dst/summary.txt" 2> /dev/null | cut -c1-200
    done
}

cmd_report() {
    local d
    printf '%-5s %-10s %-6s %-6s %-6s %-6s %s\n' phone run abi kat qmm llm "llm runs (api prompt/gen t/s)"
    for d in $FLEET; do
        local r; r=$(ls -d "$EV/$d"/*/ 2> /dev/null | sort | tail -1)
        if [ -z "$r" ] || [ ! -f "$r/summary.txt" ]; then printf '%-5s %s\n' "$d" "not run"; continue; fi
        local s=$r/summary.txt
        st() { grep -m1 "^CHECK $1 " "$s" | awk '{print $3}'; }
        local runs; runs=$(grep -E '^  run (dec|pp)_' "$s" | sed -E 's/^  run ([a-z_]+) rc=[0-9]+ api=([a-z]+) prompt_tps=([^ ]+) gen_tps=([^ ]+).*/\1:\2 \3\/\4/' | paste -sd' ' -)
        printf '%-5s %-10s %-6s %-6s %-6s %-6s %s\n' "$d" "$(basename "$r" | cut -c5-13)" "$(st abi)" "$(st kat)" "$(st qmm)" "$(st llm)" "$runs"
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
    report) cmd_report ;;
    go) cmd_go "$@" ;;
    *) sed -n '4,20p' "$0"; exit 2 ;;
esac
