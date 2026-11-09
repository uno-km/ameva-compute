# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""release_ab.py EVIDENCE_DIR [OUT_DIR]: published release against the current plugin, per phone and model.

Input: EVIDENCE_DIR/<dev>/<model>.log from scripts/termux/lab/ladder_run.sh with LADDER_VARIANTS naming
v281 (the release's backend, Vulkan), cur (current, Vulkan), optionally curcl (current, OpenCL) and cpu.
Runs are filtered as in ladder_fit.py (rc=0, valid=1, the highest clock cap and the widest core set of the phone).
Prints a Markdown table of median t/s and of the change against v281, writes release_ab.md, release_ab.json and
release_ab.png. A variant with fewer than MIN_ROUNDS valid runs is printed in parentheses and left out of the chart.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder_fit import MODELS, load  # noqa: E402

MIN_ROUNDS = 3
VARIANTS = ("v281", "cur", "curcl", "cpu")


def med(runs, key):
    vals = [r[key] for r in runs if r[key] is not None]
    return statistics.median(vals) if vals else None


def main():
    ev = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ev
    out.mkdir(parents=True, exist_ok=True)
    runs = [r for r in load(ev, core_scope="model") if r["rc"] == 0 and r["valid"] == 1]
    res = {}
    md = ["# Release 2.8.1 against the current plugin (median t/s, same session, rotated rounds)", ""]
    for dev in sorted({r["dev"] for r in runs}):
        R = [r for r in runs if r["dev"] == dev]
        md += [f"## {dev} (cores {', '.join(sorted({r['cpus'] for r in R}))})", "",
               "| model | phase | 2.8.1 Vulkan | current Vulkan | change | current OpenCL | change | CPU | runs (v281/cur/cl/cpu) | text 2.8.1 = current |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        res[dev] = {}
        for model, (pb, _) in sorted(MODELS.items(), key=lambda kv: kv[1][0]):
            by = {v: [r for r in R if r["model"] == model and r["side"] == v] for v in VARIANTS}
            if not by["v281"]:
                continue
            same = sorted({r["text"] for r in by["v281"]}) == sorted({r["text"] for r in by["cur"]})
            res[dev][model] = {"params_b": pb, "same_text": same, "runs": {v: len(by[v]) for v in VARIANTS}}
            for key, phase in (("pp", "prompt"), ("tg", "generation")):
                m = {v: med(by[v], key) if by[v] else None for v in VARIANTS}
                res[dev][model][key] = m
                fmt = lambda v: "-" if m[v] is None else (f"{m[v]:.2f}" if len(by[v]) >= MIN_ROUNDS else f"({m[v]:.2f})")
                chg = lambda v: "-" if m[v] is None or not m["v281"] else f"{(m[v] / m['v281'] - 1) * 100:+.0f} %"
                n = "/".join(str(len(by[v])) for v in VARIANTS)
                md.append(f"| {model} | {phase} | {fmt('v281')} | {fmt('cur')} | {chg('cur')} | {fmt('curcl')} | "
                          f"{chg('curcl')} | {fmt('cpu')} | {n} | {'yes' if same else 'no'} |")
        md.append("")
    plot(res, out)
    md += ["![release against current](release_ab.png)", ""]
    (out / "release_ab.json").write_text(json.dumps(res, indent=1))
    (out / "release_ab.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    short = {"qwen2.5-0.5b-instruct-q4_k_m": "Qw0.5", "Llama-3.2-1B-Instruct-Q4_K_M": "Ll1",
             "qwen2.5-1.5b-instruct-q4_k_m": "Qw1.5", "gemma-2-2b-it-Q4_K_M": "Ge2",
             "qwen2.5-3b-instruct-q4_k_m": "Qw3", "Llama-3.2-3B-Instruct-Q4_K_M": "Ll3"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax, key, title in ((axes[0], "pp", "prompt"), (axes[1], "tg", "generation")):
        labels, a, b = [], [], []
        for dev, models in sorted(res.items()):
            for model, d in sorted(models.items(), key=lambda kv: kv[1]["params_b"]):
                base = d[key]["v281"]
                if not base:
                    continue
                labels.append(f"{dev}\n{short[model]}")
                a.append(d[key]["cur"] / base if d[key]["cur"] and d["runs"]["cur"] >= MIN_ROUNDS else np.nan)
                b.append(d[key]["curcl"] / base if d[key]["curcl"] and d["runs"]["curcl"] >= MIN_ROUNDS else np.nan)
        x = np.arange(len(labels))
        ax.bar(x - 0.2, a, 0.4, color="#1f6fb2", label="current Vulkan / 2.8.1")
        ax.bar(x + 0.2, b, 0.4, color="#e3a33b", label="current OpenCL / 2.8.1")
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("speed relative to 2.8.1")
        ax.set_title(f"{title}: above 1 is faster than the released 2.8.1")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "release_ab.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
