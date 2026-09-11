import subprocess
import re
import json
import numpy as np

base_cmd = ["ssh", "-p", "8022", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "u0_a30@100.106.251.21"]

runs = 5
results = []
prompt = "The AMEVA text to speech pipeline is running."

print("================================================================================")
print("AMEVA TTS STAGE-BY-STAGE 5-RUN INFERENCE PROFILING ON GALAXY A35")
print("================================================================================")
print(f"Prompt: '{prompt}'")

for i in range(runs):
    remote_cmd = (
        "cd ~/ && "
        "unset AMEVA_TTS_FAIL_STAGE && "
        "/data/data/com.termux/files/home/sherpa-ncnn/build-vulkan/bin/sherpa-ncnn-offline-tts-mixed "
        "--vits-model-dir=./ncnn-vits-piper-en_US-lessac-high-fp16-patched "
        "--use-vulkan-compute=1 --num-threads=1 "
        f"--output-filename=./test_stage_run_{i}.wav "
        f"'{prompt}'"
    )
    
    proc = subprocess.run(base_cmd + [remote_cmd], capture_output=True, text=True, encoding="utf-8", check=True)
    stderr = proc.stderr
    
    # Parse stage timings
    m_load = re.search(r"MODEL_LOAD_TIME_MS=([\d\.]+)", stderr)
    m_text = re.search(r"TEXT_PROCESSING_TIME_MS=([\d\.]+)", stderr)
    m_enc = re.search(r"ENCODER_TIME_MS=([\d\.]+)", stderr)
    m_dp = re.search(r"DURATION_PREDICTOR_TIME_MS=([\d\.]+)", stderr)
    m_pa = re.search(r"PATH_ATTENTION_TIME_MS=([\d\.]+)", stderr)
    m_fl = re.search(r"FLOW_TIME_MS=([\d\.]+)", stderr)
    m_dec = re.search(r"DECODER_TIME_MS=([\d\.]+)", stderr)
    m_synth = re.search(r"TOTAL_SYNTHESIS_TIME_MS=([\d\.]+)", stderr)
    m_dur = re.search(r"Audio duration:\s*([\d\.]+)\s*s", stderr)
    m_rtf = re.search(r"Real-time factor \(RTF\):\s*([\d\./]+)\s*=\s*([\d\.]+)", stderr)
    
    load_ms = float(m_load.group(1)) if m_load else 0.0
    text_ms = float(m_text.group(1)) if m_text else 0.0
    enc_ms = float(m_enc.group(1)) if m_enc else 0.0
    dp_ms = float(m_dp.group(1)) if m_dp else 0.0
    pa_ms = float(m_pa.group(1)) if m_pa else 0.0
    fl_ms = float(m_fl.group(1)) if m_fl else 0.0
    dec_ms = float(m_dec.group(1)) if m_dec else 0.0
    synth_ms = float(m_synth.group(1)) if m_synth else 0.0
    audio_dur_s = float(m_dur.group(1)) if m_dur else 0.0
    audio_dur_ms = audio_dur_s * 1000.0
    rtf = float(m_rtf.group(2)) if m_rtf else (synth_ms / audio_dur_ms if audio_dur_ms > 0 else 0.0)
    
    run_data = {
        "run": i + 1,
        "model_load_ms": load_ms,
        "text_processing_ms": text_ms,
        "encoder_ms": enc_ms,
        "duration_predictor_ms": dp_ms,
        "path_attention_ms": pa_ms,
        "flow_ms": fl_ms,
        "decoder_ms": dec_ms,
        "total_synthesis_ms": synth_ms,
        "audio_dur_ms": audio_dur_ms,
        "rtf": rtf
    }
    results.append(run_data)
    
    print(f"\n[RUN {i+1}]")
    print(f"  MODEL_LOAD_TIME_MS:         {load_ms:.2f} ms")
    print(f"  TEXT_PROCESSING_TIME_MS:    {text_ms:.2f} ms")
    print(f"  ENCODER_TIME_MS:            {enc_ms:.2f} ms")
    print(f"  DURATION_PREDICTOR_TIME_MS: {dp_ms:.2f} ms")
    print(f"  PATH_ATTENTION_TIME_MS:     {pa_ms:.2f} ms")
    print(f"  FLOW_TIME_MS:               {fl_ms:.2f} ms")
    print(f"  DECODER_TIME_MS:            {dec_ms:.2f} ms")
    print(f"  TOTAL_SYNTHESIS_TIME_MS:    {synth_ms:.2f} ms")
    print(f"  AUDIO_DURATION_MS:          {audio_dur_ms:.2f} ms")
    print(f"  RTF:                        {rtf:.4f}")

# Cleanup remote temp files
subprocess.run(base_cmd + ["rm -f ~/test_stage_run_*.wav"], check=True)

# Calculate Medians (P50)
enc_p50 = float(np.median([r["encoder_ms"] for r in results]))
dp_p50 = float(np.median([r["duration_predictor_ms"] for r in results]))
pa_p50 = float(np.median([r["path_attention_ms"] for r in results]))
fl_p50 = float(np.median([r["flow_ms"] for r in results]))
dec_p50 = float(np.median([r["decoder_ms"] for r in results]))
synth_p50 = float(np.median([r["total_synthesis_ms"] for r in results]))
text_p50 = float(np.median([r["text_processing_ms"] for r in results]))
load_p50 = float(np.median([r["model_load_ms"] for r in results]))
rtf_p50 = float(np.median([r["rtf"] for r in results]))

print("\n" + "="*80)
print("FINAL STAGE MEDIAN (P50) SCORECARD:")
print("="*80)
print(f"MODEL_LOAD_TIME_P50_MS:             {load_p50:.2f} ms")
print(f"TEXT_PROCESSING_TIME_P50_MS:        {text_p50:.2f} ms")
print(f"ENCODER_TIME_P50_MS:                {enc_p50:.2f} ms ({enc_p50/synth_p50*100:.1f}%)")
print(f"DURATION_PREDICTOR_TIME_P50_MS:     {dp_p50:.2f} ms ({dp_p50/synth_p50*100:.1f}%)")
print(f"PATH_ATTENTION_TIME_P50_MS:         {pa_p50:.2f} ms ({pa_p50/synth_p50*100:.1f}%)")
print(f"FLOW_TIME_P50_MS:                   {fl_p50:.2f} ms ({fl_p50/synth_p50*100:.1f}%)")
print(f"DECODER_TIME_P50_MS:                {dec_p50:.2f} ms ({dec_p50/synth_p50*100:.1f}%)")
print(f"TOTAL_SYNTHESIS_TIME_P50_MS:        {synth_p50:.2f} ms")
print(f"RTF_P50:                            {rtf_p50:.4f}")
print("="*80)

with open("tts_stage_profiling_results.json", "w") as f:
    json.dump({
        "raw_runs": results,
        "medians": {
            "model_load_ms": load_p50,
            "text_processing_ms": text_p50,
            "encoder_ms": enc_p50,
            "duration_predictor_ms": dp_p50,
            "path_attention_ms": pa_p50,
            "flow_ms": fl_p50,
            "decoder_ms": dec_p50,
            "total_synthesis_ms": synth_p50,
            "rtf": rtf_p50
        }
    }, f, indent=2)
print("Saved raw results to tts_stage_profiling_results.json")
