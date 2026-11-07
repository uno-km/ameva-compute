#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""hw_study_summary.py RESULT_DIR [...]: tables from the output of scripts/termux/lab/hw_study_suite.sh.

Per directory: the SUITE lines (stages, skips, failures), the unit throughputs of cl_hw_bench, and per API and shape
the median kernel time over the valid rounds of every ameva_mmq_bench variant. A run counts when its cap stayed at the
target (valid=1), it ran the kernel family it was meant to (path == expect) and its sampled outputs passed (check=PASS);
the runs that do not count are listed with the reason, never folded into a figure. The llm stage gives the median
prompt and generation t/s of the valid runs per variant and the distinct generated texts.
"""
import csv
import os
import re
import statistics
import sys


def med(xs):
    return statistics.median(xs) if xs else None


def summarize(d):
    print(f"## {d}")
    log = os.path.join(d, "suite.log")
    if os.path.exists(log):
        for line in open(log, encoding="utf-8", errors="replace"):
            if line.startswith(("SUITE_INFO", "SUITE_NOTE", "SUITE_FAIL", "SUITE_SKIP", "SUITE qmm", "SUITE hw", "SUITE_END",
                                "SUITE mm_probe", "SUITE llm")):
                print("  " + line.rstrip())
    hw = os.path.join(d, "hw.log")
    if os.path.exists(hw):
        print("\n### unit throughputs (cl_hw_bench)")
        for line in open(hw, encoding="utf-8", errors="replace"):
            if line.startswith(("HW_DEVICE", "HW_SKIP")):
                print("  " + line.rstrip())
            elif line.startswith("HW "):
                print("  " + re.sub(r" (GB/s|ms|lane_loads/clk/core)=[^ ]+", "", line.rstrip()))
    tsv = os.path.join(d, "mm.tsv")
    if os.path.exists(tsv):
        rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
        print("\n### prompt MUL_MAT kernels (ameva_mmq_bench, median ms over valid rounds)")
        groups = {}
        bad = []
        for r in rows:
            key = (r["api"], r["type"], r["M"], r["N"], r["K"])
            ok = r["valid"] == "1" and r["path"] == r["expect"] and r["check"] == "PASS" and r["rc"] == "0"
            if ok:
                groups.setdefault(key, {}).setdefault(r["variant"], []).append(float(r["median_ms"]))
            else:
                bad.append(r)
        for key, vs in groups.items():
            api, t, m, n, k = key
            gop = 2.0 * int(m) * int(n) * int(k) / 1e9
            cells = sorted(((med(x), v, len(x)) for v, x in vs.items()), key=lambda c: c[0])
            ref = med(vs["mmq"]) if "mmq" in vs else None
            best_b = next((c for c in cells if c[1].startswith("mmqb_")), None)
            line = "  ".join(f"{v} {ms:.1f}({gop / ms * 1e3:.0f})x{nr}" for ms, v, nr in cells)
            note = ""
            if ref and best_b:
                note = f"  best mmqb {best_b[1][5:]} = {ref / best_b[0]:.2f}x ameva_mmq"
            print(f"  {api} {t} {m}x{n}x{k}: {line}{note}")
        if bad:
            print(f"  not counted: {len(bad)} runs")
            for r in bad[:12]:
                why = []
                if r["valid"] != "1":
                    why.append(f"cap {r['cap0']}->{r['cap1']}")
                if r["path"] != r["expect"]:
                    why.append(f"path {r['path']} != {r['expect']}")
                if r["check"] != "PASS":
                    why.append(f"check {r['check']}")
                if r["rc"] != "0":
                    why.append(f"rc {r['rc']}")
                print(f"    {r['api']} {r['type']} {r['N']}x{r['K']} {r['variant']} round {r['round']}: {', '.join(why)}")
    llm = os.path.join(d, "llm.log")
    if os.path.exists(llm):
        print("\n### model (llm_plugin_ab.sh, valid runs)")
        per = {}
        for line in open(llm, encoding="utf-8", errors="replace"):
            if not line.startswith("LAB round="):
                continue
            f = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
            if f.get("valid") != "1" or f.get("pp_tps", "NA") == "NA":
                continue
            p = per.setdefault(f["var"], {"pp": [], "tg": [], "text": set()})
            p["pp"].append(float(f["pp_tps"]))
            p["tg"].append(float(f["tg_tps"]))
            p["text"].add(f["text"])
        for v, p in per.items():
            print(f"  {v}: prompt {med(p['pp']):.2f} t/s, generation {med(p['tg']):.2f} t/s, n={len(p['pp'])}, "
                  f"texts={','.join(sorted(p['text']))}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for a in sys.argv[1:]:
        summarize(a)
