#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# tts_sherpa_baseline.py MODEL_DIR OUT.wav "text" [--threads N] [--reps R]
# The CPU product path that termux-tts uses for Kokoro today: sherpa-onnx with the int8 model (its own espeak-ng
# G2P), text to WAV. Prints the synthesis time per repetition and the real-time factor; writes the WAV to disk only.
import sys
import time

import sherpa_onnx


def main():
    args = sys.argv[1:]
    threads, reps = 4, 3
    if "--threads" in args:
        i = args.index("--threads")
        threads = int(args[i + 1])
        del args[i:i + 2]
    if "--reps" in args:
        i = args.index("--reps")
        reps = int(args[i + 1])
        del args[i:i + 2]
    if len(args) != 3:
        sys.exit("usage: tts_sherpa_baseline.py MODEL_DIR OUT.wav TEXT [--threads N] [--reps R]")
    d, out, text = args
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(model=d + "/model.int8.onnx", voices=d + "/voices.bin",
                                                          tokens=d + "/tokens.txt", data_dir=d + "/espeak-ng-data"),
            num_threads=threads, provider="cpu"),
        max_num_sentences=100)
    tts = sherpa_onnx.OfflineTts(cfg)
    times = []
    audio = None
    for _ in range(reps):
        t0 = time.perf_counter()
        audio = tts.generate(text, sid=0, speed=1.0)
        times.append(time.perf_counter() - t0)
    secs = len(audio.samples) / audio.sample_rate
    sherpa_onnx.write_wave(out, audio.samples, audio.sample_rate)
    print(f"sherpa-onnx kokoro int8 cpu threads {threads}: audio {secs:.2f} s at {audio.sample_rate} Hz, "
          f"ms min {min(times) * 1000:.1f} med {sorted(times)[len(times) // 2] * 1000:.1f}, RTF {min(times) / secs:.3f}")


if __name__ == "__main__":
    main()
