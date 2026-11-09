#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""Every kernel variant of the product as SPIR-V files and OpenCL C files, for driver-side inspection (research handover
2026-11-09, R1/R2: re_dump vk|cl on the phones).

usage: kernel_survey.py OUT_DIR [build_ameva_kernels.py options]
Runs scripts/build_ameva_kernels.py unchanged (headers go to OUT_DIR/*.h, not into the tree) and copies each module it
compiles to OUT_DIR/spv/<n>_<basename>.spv, with OUT_DIR/index.tsv: number, source, defines.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
import build_ameva_kernels as b  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = os.path.abspath(sys.argv[1])
    os.makedirs(os.path.join(out, "spv"), exist_ok=True)
    index = open(os.path.join(out, "index.tsv"), "w")
    original = b.compile_module
    count = [0]

    def recording(args, src, defines, spv, cl, *rest, **kw):
        result = original(args, src, defines, spv, cl, *rest, **kw)
        count[0] += 1
        name = "%03d_%s" % (count[0], os.path.basename(spv))
        shutil.copyfile(spv, os.path.join(out, "spv", name))
        index.write("%s\t%s\t%s\n" % (name, os.path.basename(src), " ".join(defines)))
        return result

    b.compile_module = recording
    sys.argv = [sys.argv[0], "--out", os.path.join(out, "kernels_spv.h"), "--cl-out", os.path.join(out, "kernels_cl.h")] + sys.argv[2:]
    b.main()
    index.close()
    print("kernel_survey: %d modules in %s" % (count[0], out))


if __name__ == "__main__":
    main()
