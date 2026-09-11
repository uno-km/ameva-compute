import subprocess
import json

stages = ["context", "kernel", "buffer", "dispatch", "wait", "readback"]
results = []

test_wav = "./test_fail_out.wav"
base_cmd = [
    "ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21"
]

print("================================================================================")
print("AMEVA TTS MIXED ADAPTER FAILURE INJECTION TEST RUNNER")
print("================================================================================")

for stage in stages:
    print(f"\n>>> TESTING GATE: AMEVA_TTS_FAIL_STAGE={stage}")
    # Ensure test_wav does not exist
    subprocess.run(base_cmd + [f"rm -f {test_wav}"], check=True)
    
    # Run mixed binary with failure injection
    remote_cmd = (
        f"cd ~/ && "
        f"export AMEVA_TTS_FAIL_STAGE={stage} && "
        f"./sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts-mixed "
        f"--vits-model-dir=./ncnn-vits-piper-en_US-lessac-high-fp16-patched "
        f"--use-vulkan-compute=1 --num-threads=1 "
        f"--output-filename={test_wav} "
        f"'The AMEVA text to speech pipeline is running.'"
    )
    
    proc = subprocess.run(base_cmd + [remote_cmd], capture_output=True, text=True)
    exit_code = proc.returncode
    stdout = proc.stdout
    stderr = proc.stderr
    combined = stdout + "\n" + stderr
    
    # Check if test_wav exists
    check_wav = subprocess.run(base_cmd + [f"[ -f {test_wav} ] && echo EXISTS || echo NOT_EXISTS"], capture_output=True, text=True)
    wav_exists = "EXISTS" in check_wav.stdout and "NOT_EXISTS" not in check_wav.stdout
    
    # Analyze criteria
    process_exit_non_zero = (exit_code != 0)
    output_success_not_reported = ("Saved to" not in combined)
    partial_wav_removed = (not wav_exists)
    custom_layer_exec = ("CUSTOM_LAYER_EXECUTION_COUNT=" in stderr and "CUSTOM_LAYER_EXECUTION_COUNT=0" not in stderr) or ("FORWARD(Mat) INVOKED" in stderr)
    failure_count_1 = ("AMEVA_GPU_FAILURE_COUNT=1" in stderr)
    silent_fallback_0 = ("SILENT_FALLBACK_COUNT=0" in stderr)
    ctx_balanced = ("CONTEXT_LIFECYCLE_BALANCED=true" in stderr)
    kern_balanced = ("KERNEL_LIFECYCLE_BALANCED=true" in stderr)
    buf_balanced = ("BUFFER_LIFECYCLE_BALANCED=true" in stderr)
    
    all_pass = (
        process_exit_non_zero and
        output_success_not_reported and
        partial_wav_removed and
        custom_layer_exec and
        failure_count_1 and
        silent_fallback_0 and
        ctx_balanced and
        kern_balanced and
        buf_balanced
    )
    
    res = {
        "stage": stage,
        "exit_code": exit_code,
        "process_exit_non_zero": process_exit_non_zero,
        "output_success_not_reported": output_success_not_reported,
        "partial_wav_removed": partial_wav_removed,
        "custom_layer_exec": custom_layer_exec,
        "failure_count_1": failure_count_1,
        "silent_fallback_0": silent_fallback_0,
        "ctx_balanced": ctx_balanced,
        "kern_balanced": kern_balanced,
        "buf_balanced": buf_balanced,
        "gate_pass": all_pass
    }
    results.append(res)
    
    print(f"  EXIT_CODE: {exit_code} (NON_ZERO: {process_exit_non_zero})")
    print(f"  OUTPUT_SUCCESS_NOT_REPORTED: {output_success_not_reported}")
    print(f"  PARTIAL_WAV_REMOVED: {partial_wav_removed}")
    print(f"  CUSTOM_LAYER_EXECUTION: {custom_layer_exec}")
    print(f"  AMEVA_GPU_FAILURE_COUNT=1: {failure_count_1}")
    print(f"  SILENT_FALLBACK_COUNT=0: {silent_fallback_0}")
    print(f"  CONTEXT_LIFECYCLE_BALANCED: {ctx_balanced}")
    print(f"  KERNEL_LIFECYCLE_BALANCED: {kern_balanced}")
    print(f"  BUFFER_LIFECYCLE_BALANCED: {buf_balanced}")
    print(f"  --> GATE STATUS: {'PASS' if all_pass else 'FAIL'}")

print("\n================================================================================")
print("FINAL FAILURE INJECTION SUMMARY:")
passed_gates = sum(1 for r in results if r["gate_pass"])
print(f"MIXED_ADAPTER_FAILURE_INJECTION_GATES={passed_gates}/6")
partial_wav_gates = sum(1 for r in results if r["partial_wav_removed"])
print(f"PARTIAL_WAV_REMOVAL_GATES={partial_wav_gates}/6")
lifecycle_gates = sum(1 for r in results if r["ctx_balanced"] and r["kern_balanced"] and r["buf_balanced"])
print(f"LIFECYCLE_BALANCE_GATES={lifecycle_gates}/6")
print("================================================================================")

with open("mixed_failure_injection_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to mixed_failure_injection_results.json")