#!/usr/bin/env python3
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""Table of scripts/termux/lab/matrix_run.sh logs: one row per phone, model and variant.

usage: matrix_table.py DEV=LOG [DEV=LOG ...]
Prompt and generation t/s are medians over the runs that ended with rc=0 and valid=1 (clock cap unchanged). "valid"
counts those runs out of all runs of the variant. "gen" is the number of distinct generated texts among the rc=0 runs,
and "nonascii" the largest count of non-ASCII bytes in a generated text (corrupted output shows here). A variant with
no rc=0 run prints the last failure.
"""
import re
import statistics
import sys


def parse(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("MX model="):
                continue
            d = dict(m.groups() for m in re.finditer(r"(\w+)=(\S*)", line))
            rows.append(d)
    return rows


def med(values):
    return "%.1f" % statistics.median(values) if values else "-"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("| phone | model | variant | prompt t/s | gen t/s | valid | texts | nonascii | cpus | note |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arg in sys.argv[1:]:
        dev, path = arg.split("=", 1)
        rows = parse(path)
        keys = []
        for r in rows:
            k = (r.get("model", ""), r.get("var", ""))
            if k not in keys:
                keys.append(k)
        for model, var in keys:
            rs = [r for r in rows if r.get("model") == model and r.get("var") == var]
            ok = [r for r in rs if r.get("rc") == "0"]
            good = [r for r in ok if r.get("valid") == "1"]
            def num(key):
                return [float(r[key]) for r in good if r.get(key) not in (None, "", "NA")]
            note = ""
            if not ok:
                note = "FAIL rc=%s %s" % (rs[-1].get("rc"), rs[-1].get("err", ""))
            cpus = sorted({r.get("cpus", "") for r in rs})
            print("| %s | %s | %s | %s | %s | %d/%d | %d | %s | %s | %s |" % (
                dev, model, var, med(num("pp_tps")), med(num("tg_tps")), len(good), len(rs),
                len({r.get("gen") for r in ok}), max([int(r.get("nonascii", 0)) for r in ok], default=0),
                ",".join(cpus), note))


if __name__ == "__main__":
    main()
