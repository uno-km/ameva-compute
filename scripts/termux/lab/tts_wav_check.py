#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# tts_wav_check.py FILE.wav [REFERENCE.wav|REFERENCE.f32 ...]
# Inspects a synthesized WAV without playing it: RIFF/WAVE header (format, channels, sample rate, bits, data size),
# duration, RMS and peak, share of silent 20 ms frames, and, per reference, log-mel distance (80 mel bands, 1024-point
# frames, hop 256; mean over frames of the RMS over bands of the difference in dB), spectral-envelope correlation
# and waveform SNR. A .f32 reference is raw little-endian float32 at the WAV's rate. Needs numpy only.
import struct
import sys

import numpy as np


def read_wav(path):
    b = open(path, "rb").read()
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    i, fmt, data = 12, None, None
    while i + 8 <= len(b):
        cid, size = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        body = b[i + 8:i + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            data = body
        i += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError(f"{path}: fmt or data chunk missing")
    tag, ch, sr, _, _, bits = fmt
    if tag != 1 or bits != 16:
        raise ValueError(f"{path}: format {tag} with {bits} bits (16-bit PCM expected)")
    x = np.frombuffer(data, "<i2").astype(np.float64) / 32768.0
    return {"channels": ch, "rate": sr, "bits": bits, "frames": len(x) // ch}, x.reshape(-1, ch)[:, 0]


def mel_fb(sr, n_fft=1024, n_mels=80):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        for k in range(l, c):
            fb[m, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m, k] = (r - k) / max(r - c, 1)
    return fb


def logmel(x, sr):
    n_fft, hop = 1024, 256
    x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    fr = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
    p = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    return 10 * np.log10(np.maximum(p @ mel_fb(sr, n_fft).T, 1e-10))


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: tts_wav_check.py FILE.wav [REFERENCE.wav|.f32 ...]")
    info, x = read_wav(sys.argv[1])
    sr = info["rate"]
    fr = x[: len(x) // (sr // 50) * (sr // 50)].reshape(-1, sr // 50)
    silent = float(np.mean(np.sqrt(np.mean(fr ** 2, axis=1)) < 1e-3)) if len(fr) else 1.0
    print(f"{sys.argv[1]}: PCM {info['bits']}-bit, {info['channels']} ch, {sr} Hz, {info['frames']} frames "
          f"({info['frames'] / sr:.3f} s), rms {np.sqrt(np.mean(x ** 2)):.4f}, peak {np.abs(x).max():.4f}, "
          f"silent 20 ms frames {silent:.2%}")
    mx = logmel(x, sr)
    for ref in sys.argv[2:]:
        y = read_wav(ref)[1] if ref.endswith(".wav") else np.fromfile(ref, "<f4").astype(np.float64)
        n = min(len(x), len(y))
        my = logmel(y[:n], sr)
        m = min(len(mx), len(my))
        d = mx[:m] - my[:m]
        lsd = float(np.mean(np.sqrt(np.mean(d ** 2, axis=1))))
        corr = float(np.corrcoef(mx[:m].ravel(), my[:m].ravel())[0, 1])
        snr = 10 * np.log10(np.sum(y[:n] ** 2) / max(np.sum((x[:n] - y[:n]) ** 2), 1e-30))
        print(f"  vs {ref}: samples {len(x)} / {len(y)}, log-mel distance {lsd:.3f} dB, log-mel correlation {corr:.5f}, "
              f"waveform SNR {snr:.2f} dB")


if __name__ == "__main__":
    main()
