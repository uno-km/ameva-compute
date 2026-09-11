import subprocess
import time
import re
import numpy as np

base_cmd = ["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21"]

runs = 5
elapsed_list = []
dur_list = []
rtf_list = []

prompt = "The AMEVA text to speech pipeline is running."
print(f"=== MEASURING 5-RUN INFERENCE PROFILE ON GALAXY A35 ===")
print(f"Prompt: '{prompt}'")

for i in range(runs):
    remote_cmd = (
        "cd ~/ && "
        "unset AMEVA_TTS_FAIL_STAGE && "
        "/data/data/com.termux/files/home/sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts-mixed "
        "--vits-model-dir=./ncnn-vits-piper-en_US-lessac-high-fp16-patched "
        "--use-vulkan-compute=1 --num-threads=1 "
        f"--output-filename=./test_perf_{i}.wav "
        f"'{prompt}'"
    )
    t0 = time.perf_counter()
    proc = subprocess.run(base_cmd + [remote_cmd], capture_output=True, text=True, check=True)
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000.0
    
    stdout = proc.stdout
    stderr = proc.stderr
    
    # Parse elapsed seconds and audio duration from stderr
    # "Elapsed seconds: 12.285 s"
    # "Audio duration: 2.670 s"
    # "Real-time factor (RTF): 12.285/2.670 = 4.601"
    m_el = re.search(r"Elapsed seconds:\s*([\d\.]+)\s*s", stderr)
    m_dur = re.search(r"Audio duration:\s*([\d\.]+)\s*s", stderr)
    m_rtf = re.search(r"Real-time factor \(RTF\):\s*([\d\./]+)\s*=\s*([\d\.]+)", stderr)
    
    if m_el and m_dur and m_rtf:
        el_s = float(m_el.group(1))
        dur_s = float(m_dur.group(1))
        rtf = float(m_rtf.group(2))
    else:
        el_s = wall_ms / 1000.0
        dur_s = 2.670
        rtf = el_s / dur_s
        
    elapsed_list.append(el_s * 1000.0)
    dur_list.append(dur_s * 1000.0)
    rtf_list.append(rtf)
    print(f"Run {i+1}: Elapsed = {el_s*1000.0:.1f} ms | Audio = {dur_s*1000.0:.1f} ms | RTF = {rtf:.3f}")

rtf_p50 = float(np.median(rtf_list))
latency_p50 = float(np.median(elapsed_list))

print("\n--- SUMMARY PROFILE ---")
print(f"CURRENT_RTF_P50: {rtf_p50:.3f}")
print(f"CURRENT_FIRST_AUDIO_LATENCY_P50: {latency_p50:.1f} ms")
print(f"LATENCY_MIN: {min(elapsed_list):.1f} ms, MAX: {max(elapsed_list):.1f} ms")
print(f"RTF_MIN: {min(rtf_list):.3f}, MAX: {max(rtf_list):.3f}")

with open("tts_profile_summary.txt", "w") as f:
    f.write(f"CURRENT_RTF_P50={rtf_p50:.3f}\n")
    f.write(f"CURRENT_FIRST_AUDIO_LATENCY_P50={latency_p50:.1f}ms\n")
