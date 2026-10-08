#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""make_test_lora.py OUT.safetensors [rank] [scale] [seed]

A synthetic LoRA for an SD 1.x UNet in the kohya naming stable-diffusion.cpp reads: lora_down / lora_up / alpha for
the self-attention projections (attn1 to_q, to_k, to_v, to_out.0) of every transformer block. The values are
pseudo-random (a fixed generator, so every device writes the same file); it is a test input that changes the image,
not a trained adapter. No dependency beyond the standard library.

rank defaults to 4, scale (standard deviation-like amplitude of both matrices) to 0.08, seed to 1.
Prints the file's SHA-256 and tensor count.
"""
import hashlib
import json
import struct
import sys

# (block prefix, attention modules, channels) of the SD 1.x UNet
BLOCKS = [("down_blocks_0", 2, 320), ("down_blocks_1", 2, 640), ("down_blocks_2", 2, 1280), ("mid_block", 1, 1280),
          ("up_blocks_1", 3, 1280), ("up_blocks_2", 3, 640), ("up_blocks_3", 3, 320)]
PROJ = ["attn1_to_q", "attn1_to_k", "attn1_to_v", "attn1_to_out_0"]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08
    state = (int(sys.argv[4]) if len(sys.argv) > 4 else 1) & 0xFFFFFFFF or 1

    def values(n):
        nonlocal state
        buf = bytearray(4 * n)
        for i in range(n):
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFFFFFF
            struct.pack_into("<f", buf, 4 * i, ((state >> 8) * (2.0 / 16777216.0) - 1.0) * scale)
        return bytes(buf)

    header, chunks, off = {}, [], 0

    def add(name, shape, data):
        nonlocal off
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(data)]}
        chunks.append(data)
        off += len(data)

    for prefix, n_attn, ch in BLOCKS:
        for a in range(n_attn):
            for p in PROJ:
                base = "lora_unet_%s_attentions_%d_transformer_blocks_0_%s" % (prefix, a, p)
                add(base + ".lora_down.weight", [rank, ch], values(rank * ch))
                add(base + ".lora_up.weight", [ch, rank], values(ch * rank))
                add(base + ".alpha", [], struct.pack("<f", float(rank)))
    head = json.dumps(header, separators=(",", ":")).encode("utf-8")
    head += b" " * (-len(head) % 8)
    blob = struct.pack("<Q", len(head)) + head + b"".join(chunks)
    with open(out, "wb") as f:
        f.write(blob)
    print("wrote %s: %d tensors, rank %d, scale %g, sha256 %s" % (out, len(header), rank, scale,
                                                                 hashlib.sha256(blob).hexdigest()))


if __name__ == "__main__":
    main()
