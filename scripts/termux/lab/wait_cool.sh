#!/data/data/com.termux/files/usr/bin/bash
# wait_cool.sh [MAX_MINUTES] [TARGET_HZ]
# Waits until the Adreno thermal cap (/sys/class/kgsl/kgsl-3d0/max_gpuclk) is back at TARGET_HZ (default: the largest
# value seen in the available frequency list, else 587000000) or MAX_MINUTES (default 20) pass. Prints the state.
set -u
F=/sys/class/kgsl/kgsl-3d0/max_gpuclk
[ -r "$F" ] || { echo "cool: no readable $F (not Adreno); nothing to wait for"; exit 0; }
LIM=${1:-20}
T=${2:-}
if [ -z "$T" ]; then
    T=$(tr ' ' '\n' < /sys/class/kgsl/kgsl-3d0/gpu_available_frequencies 2>/dev/null | sort -n | tail -1)
    [ -n "$T" ] || T=587000000
fi
for i in $(seq 0 $((LIM * 2))); do
    c=$(cat "$F")
    if [ "$c" -ge "$T" ]; then echo "cool: max_gpuclk=$c after $((i * 30)) s"; exit 0; fi
    sleep 30
done
echo "cool: still capped at $(cat "$F") Hz (target $T) after $LIM min, gpu temp $(cat /sys/class/kgsl/kgsl-3d0/temp 2>/dev/null)"
exit 1
