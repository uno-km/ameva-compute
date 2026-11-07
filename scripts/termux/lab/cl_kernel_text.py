#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""cl_kernel_text.py HEADER ARRAY OUT: write the OpenCL C text of one kernel of ggml_ameva_kernels_cl.h.

ARRAY is the C array name (for example kAmevaClMmqb_1). The file OUT can be edited and given to the engine with
AMEVA_CL_SOURCE_DIR=<dir> as <dir>/<kernel name>.cl (the engine announces every replacement), which is how kernel
ablations run in the product without a rebuild (ameva-foundation N-20261107-hw-bottleneck).
"""
import re
import sys


def unescape(lit):
    out, i = [], 0
    simple = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'", "r": "\r", "0": "\0"}
    while i < len(lit):
        c = lit[i]
        if c == "\\":
            n = lit[i + 1]
            if n == "x":
                m = re.match(r"[0-9a-fA-F]+", lit[i + 2:])
                out.append(chr(int(m.group(0), 16)))
                i += 2 + len(m.group(0))
                continue
            if n not in simple:
                sys.exit("cl_kernel_text: unknown escape \\%s" % n)
            out.append(simple[n])
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    header, name, dst = sys.argv[1:]
    text = open(header, encoding="utf-8").read()
    m = re.search(r"static const char %s\[\] =\s*(.*?);\s*\n" % re.escape(name), text, re.S)
    if not m:
        sys.exit("cl_kernel_text: %s not found in %s" % (name, header))
    lits = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    body = "".join(unescape(s) for s in lits)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    print("cl_kernel_text: %s -> %s (%d bytes)" % (name, dst, len(body)))


if __name__ == "__main__":
    main()
