#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""Driver statistics of every kernel module (re_dump vk output per module) as one CSV, plus the modules the driver
spills to memory.

usage: kernel_survey_table.py INDEX.tsv STAT_DIR OUT.csv
STAT_DIR holds <module>.txt files (re_dump output, last line rc=N). Spill: Adreno "Scratch memory usage per shader
instance" > 0, or Turnip "STP Count" / "LDP Count" > 0 (private memory stores / loads). Modules whose compile failed
are listed with their rc.
"""
import csv
import os
import re
import sys


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    index, stat_dir, out = sys.argv[1:]
    rows, names = [], []
    for line in open(index, encoding="utf-8"):
        mod, src, defs = (line.rstrip("\n").split("\t") + ["", ""])[:3]
        path = os.path.join(stat_dir, mod + ".txt")
        row = {"module": mod, "source": src, "defines": defs}
        if not os.path.exists(path):
            row["rc"] = "missing"
        else:
            for l in open(path, encoding="utf-8", errors="replace"):
                m = re.match(r"stat\[0\] (.+?) = (\S+)", l)
                if m:
                    row[m.group(1)] = m.group(2)
                    if m.group(1) not in names:
                        names.append(m.group(1))
                m = re.match(r"(compile_ms|rc)=(\S+)", l.strip())
                if m:
                    row[m.group(1)] = m.group(2)
        rows.append(row)
    cols = ["module", "source", "defines", "rc", "compile_ms"] + names
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def num(r, k):
        try:
            return float(r.get(k, 0) or 0)
        except ValueError:
            return 0.0

    failed = [r for r in rows if r.get("rc") != "0"]
    spill = [r for r in rows if num(r, "Scratch memory usage per shader instance") > 0 or num(r, "STP Count") > 0
             or num(r, "LDP Count") > 0]
    print("modules=%d failed=%d spilling=%d -> %s" % (len(rows), len(failed), len(spill), out))
    for r in failed:
        print("FAILED %s %s %s rc=%s" % (r["module"], r["source"], r["defines"], r.get("rc")))
    for r in spill:
        print("SPILL %s %s scratch=%s stp=%s ldp=%s regs=%s" % (
            r["module"], r["defines"][:70], r.get("Scratch memory usage per shader instance", "-"), r.get("STP Count", "-"),
            r.get("LDP Count", "-"), r.get("Registers used", r.get("Full precision register footprint per shader instance", "-"))))


if __name__ == "__main__":
    main()
