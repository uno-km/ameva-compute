#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""img_psnr.py REF.png TEST.png [--min-psnr DB | --equal]

Compares two PNG images of the same size sample by sample. Reads 8-bit, non-interlaced grey, grey+alpha, RGB and RGBA
files (what sd-cli writes) with the standard library only, so it runs on the devices.

Prints one line:
  PSNR ref=<file> test=<file> size=<W>x<H>x<C> psnr_db=<dB or inf> mse=<..> max_abs=<..> differing=<n>/<samples>
Exit status: 0 compared and the gate, if one was given, holds; 1 the gate does not hold; 2 a file could not be read or
the two images differ in size or channel count.
"""
import math
import struct
import sys
import zlib

CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def read_png(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG file")
    pos, idat, header = 8, [], None
    while pos + 12 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        crc = data[pos + 8 + length:pos + 12 + length]
        if len(body) != length or len(crc) != 4:
            raise ValueError(f"{path}: truncated {kind!r} chunk")
        if zlib.crc32(kind + body) != struct.unpack(">I", crc)[0]:
            raise ValueError(f"{path}: CRC error in {kind!r} chunk")
        pos += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if header is None:
        raise ValueError(f"{path}: no IHDR chunk")
    width, height, depth, colour, _, _, interlace = header
    if depth != 8 or colour not in CHANNELS or interlace != 0:
        raise ValueError(f"{path}: unsupported PNG form (depth {depth}, colour type {colour}, interlace {interlace})")
    bpp = CHANNELS[colour]
    stride = width * bpp
    raw = zlib.decompress(b"".join(idat))
    if len(raw) != height * (stride + 1):
        raise ValueError(f"{path}: {len(raw)} bytes of image data, expected {height * (stride + 1)}")
    out = bytearray(height * stride)
    prev = bytearray(stride)
    for y in range(height):
        base = y * (stride + 1)
        kind = raw[base]
        line = bytearray(raw[base + 1:base + 1 + stride])
        if kind == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif kind == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif kind == 3:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 255
        elif kind == 4:
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                up = prev[i]
                corner = prev[i - bpp] if i >= bpp else 0
                p = left + up - corner
                dl, du, dc = abs(p - left), abs(p - up), abs(p - corner)
                pred = left if (dl <= du and dl <= dc) else (up if du <= dc else corner)
                line[i] = (line[i] + pred) & 255
        elif kind != 0:
            raise ValueError(f"{path}: unknown filter type {kind} in row {y}")
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return width, height, bpp, bytes(out)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    min_psnr, equal = None, "--equal" in argv
    if "--min-psnr" in argv:
        i = argv.index("--min-psnr")
        min_psnr = float(argv[i + 1])
        args.remove(argv[i + 1])
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        ref, test = read_png(args[0]), read_png(args[1])
    except (OSError, ValueError, zlib.error) as e:
        print(f"PSNR_ERROR {e}")
        return 2
    if ref[:3] != test[:3]:
        print(f"PSNR_ERROR size mismatch: {args[0]} {ref[0]}x{ref[1]}x{ref[2]}, {args[1]} {test[0]}x{test[1]}x{test[2]}")
        return 2
    a, b = ref[3], test[3]
    sq = max_abs = differing = 0
    for x, y in zip(a, b):
        if x != y:
            d = x - y if x > y else y - x
            sq += d * d
            differing += 1
            if d > max_abs:
                max_abs = d
    mse = sq / len(a)
    psnr = math.inf if sq == 0 else 10.0 * math.log10(255.0 * 255.0 / mse)
    print(f"PSNR ref={args[0]} test={args[1]} size={ref[0]}x{ref[1]}x{ref[2]} "
          f"psnr_db={'inf' if sq == 0 else format(psnr, '.2f')} mse={mse:.5f} max_abs={max_abs} "
          f"differing={differing}/{len(a)}")
    if equal and sq != 0:
        return 1
    if min_psnr is not None and psnr < min_psnr:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
