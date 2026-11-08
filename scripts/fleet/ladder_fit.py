# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""ladder_fit.py EVIDENCE_DIR [OUT_DIR]: size-ladder analysis of GPU vs CPU LLM speed.

Input: EVIDENCE_DIR/<dev>/<model>.log, the output of scripts/termux/lab/llm_plugin_ab.sh (lines "LAB round=..."),
as fetched from the phones by the size-ladder runs (variants "gpu" and "cpu").
Only runs with rc=0 and valid=1 are used; the core set of every run is kept, and a device whose CPU runs did not
have every core is flagged (its CPU curve is a lower bound, not a reference).

For every device, side (gpu, cpu) and phase (prompt = pp_tps, generation = tg_tps) the median t/s per model is
fitted against the model size with five two-parameter laws, and the errors are compared:
  inv-affine  1/v = a + b*P   (time per token = fixed overhead a + cost per parameter b)
  linear      v = c + d*P
  log         v = c + d*ln P
  power       v = c * P^k     (straight line in log-log)
  exp         v = c * e^(k*P)
The error is the root mean square of the relative error of v, and leave-one-out (refit without the model, predict
it) is reported next to it because with 3-6 points a two-parameter law can fit by accident.
From the inv-affine fits the crossover size where GPU and CPU take the same time is
  P* = (a_gpu - a_cpu) / (b_cpu - b_gpu)   (meaningful when the signs make it positive).
Outputs: OUT_DIR/ladder_fit.json, ladder_fit.md (tables), and PNG charts per device and phase.
"""
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

# parameters (billions, published totals) and the family of every ladder model
MODELS = {
    "qwen2.5-0.5b-instruct-q4_k_m": (0.494, "Qwen2.5"),
    "qwen2.5-1.5b-instruct-q4_k_m": (1.54, "Qwen2.5"),
    "qwen2.5-3b-instruct-q4_k_m": (3.09, "Qwen2.5"),
    "Llama-3.2-1B-Instruct-Q4_K_M": (1.24, "Llama 3.2"),
    "Llama-3.2-3B-Instruct-Q4_K_M": (3.21, "Llama 3.2"),
    "gemma-2-2b-it-Q4_K_M": (2.61, "Gemma 2"),
}
# GGUF file bytes (the weights a generated token reads, to first order), checked on the phones 2026-11-08
FILE_BYTES = {
    "qwen2.5-0.5b-instruct-q4_k_m": 491400032, "qwen2.5-1.5b-instruct-q4_k_m": 1117320736,
    "qwen2.5-3b-instruct-q4_k_m": 2104932768, "Llama-3.2-1B-Instruct-Q4_K_M": 807694464,
    "Llama-3.2-3B-Instruct-Q4_K_M": 2019377696, "gemma-2-2b-it-Q4_K_M": 1708582752,
}
# fits reported per device: (label, x of a row, which rows)
SUBSETS = (
    ("all models, x = params (B)", lambda r: r["params_b"], lambda r: True),
    ("all models, x = file (GB)", lambda r: r["file_gb"], lambda r: True),
    ("Qwen2.5 only, x = params (B)", lambda r: r["params_b"], lambda r: r["family"] == "Qwen2.5"),
)
LAB = re.compile(r"^LAB round=(\d+) .*var=(\S+) rc=(\d+) pp_tps=(\S+) tg_tps=(\S+) text=(\S+) .*valid=(\d)(.*)$")


def load(ev: Path):
    runs = []
    for dev_dir in sorted(p for p in ev.iterdir() if p.is_dir()):
        for log in sorted(dev_dir.glob("*.log")):
            model = log.stem
            if model not in MODELS:
                continue
            for line in log.read_text(errors="replace").splitlines():
                m = LAB.match(line)
                if not m:
                    continue
                rnd, var, rc, pp, tg, text, valid, rest = m.groups()
                cpus = re.search(r"cpus=(\S+)", rest)
                caps = re.findall(r"cap[01]=(\d+)", line)
                runs.append(dict(dev=dev_dir.name, model=model, round=int(rnd), side=var, rc=int(rc),
                                 pp=float(pp) if pp != "NA" else None, tg=float(tg) if tg != "NA" else None,
                                 text=text, valid=int(valid), cpus=cpus.group(1) if cpus else "?",
                                 cap=min(int(c) for c in caps) if caps else None))
    # a run is valid only at the highest clock cap seen on its device: llm_plugin_ab.sh compares against the cap at
    # its own start, which may already be a lowered one (S21 at 403 of 858 MHz passed as valid=1, 2026-11-08)
    top = {}
    for r in runs:
        if r["cap"] is not None:
            top[r["dev"]] = max(top.get(r["dev"], 0), r["cap"])
    for r in runs:
        if r["cap"] is not None and r["cap"] < top[r["dev"]]:
            r["valid"] = 0
    # the same for the core set: a phone whose Termux went to the background mid-run continues on fewer cores
    # (A53 0-7 -> 0-2, 2026-11-08), which is another condition for the CPU and for the GPU's host thread
    wide = {}
    for r in runs:
        wide[r["dev"]] = max(wide.get(r["dev"], 0), ncores(r["cpus"]))
    for r in runs:
        if ncores(r["cpus"]) < wide[r["dev"]]:
            r["valid"] = 0
    return runs


MIN_CPU_REFERENCE_CORES = 4


def ncores(s):
    """Number of cores of a Cpus_allowed_list such as 0-2,4-7 (0 when unknown)."""
    n = 0
    for part in s.split(","):
        a, _, b = part.partition("-")
        if not a.isdigit():
            return 0
        n += int(b or a) - int(a) + 1
    return n


def fit_all(P, v):
    P = np.asarray(P, float)
    v = np.asarray(v, float)
    laws = {}

    def lsq(x, y):
        A = np.vstack([np.ones_like(x), x]).T
        (c0, c1), *_ = np.linalg.lstsq(A, y, rcond=None)
        return c0, c1

    laws["inv-affine"] = (lambda x, y: lsq(x, 1.0 / y), lambda p, c: 1.0 / (c[0] + c[1] * p))
    laws["linear"] = (lambda x, y: lsq(x, y), lambda p, c: c[0] + c[1] * p)
    laws["log"] = (lambda x, y: lsq(np.log(x), y), lambda p, c: c[0] + c[1] * np.log(p))
    laws["power"] = (lambda x, y: lsq(np.log(x), np.log(y)), lambda p, c: np.exp(c[0]) * p ** c[1])
    laws["exp"] = (lambda x, y: lsq(x, np.log(y)), lambda p, c: np.exp(c[0] + c[1] * p))
    out = {}
    for name, (fit, pred) in laws.items():
        c = fit(P, v)
        rel = pred(P, c) / v - 1.0
        rms = float(np.sqrt(np.mean(rel ** 2)))
        loo = []
        if len(P) >= 4:
            for i in range(len(P)):
                keep = np.arange(len(P)) != i
                ci = fit(P[keep], v[keep])
                loo.append(float(pred(P[i:i + 1], ci)[0] / v[i] - 1.0))
        out[name] = dict(coef=[float(c[0]), float(c[1])], rms_rel=rms,
                         loo_rms_rel=float(np.sqrt(np.mean(np.square(loo)))) if loo else None)
    return out


def main():
    ev = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ev
    out.mkdir(parents=True, exist_ok=True)
    runs = [r for r in load(ev) if r["rc"] == 0 and r["valid"] == 1]
    devs = sorted({r["dev"] for r in runs})
    result = {"devices": {}}
    md = ["# Size ladder: GPU vs CPU", ""]
    for dev in devs:
        R = [r for r in runs if r["dev"] == dev]
        cpus = sorted({r["cpus"] for r in R})
        cpu_ok = max(ncores(c) for c in cpus) >= MIN_CPU_REFERENCE_CORES
        sides = ("gpu", "cpu") if cpu_ok else ("gpu",)
        dres = {"cpus": cpus, "cpu_reference": cpu_ok, "models": {}, "fits": {}, "crossover_b": {}}
        md += [f"## {dev} (core sets seen: {', '.join(cpus)})", ""]
        if not cpu_ok:
            md += [f"CPU columns are not a reference: fewer than {MIN_CPU_REFERENCE_CORES} cores. No CPU fit, ratio "
                   "or crossover is derived from them.", ""]
        md += [
               "| model | params (B) | GPU prompt | CPU prompt | GPU gen | CPU gen | gen GPU/CPU | rounds | text GPU=CPU |",
               "|---|---|---|---|---|---|---|---|---|"]
        rows = []
        for model, (pb, fam) in sorted(MODELS.items(), key=lambda kv: kv[1][0]):
            g = [r for r in R if r["model"] == model and r["side"] == "gpu"]
            c = [r for r in R if r["model"] == model and r["side"] == "cpu"]
            if not g or not c:
                continue
            med = lambda xs, k: statistics.median([x[k] for x in xs if x[k] is not None])
            row = dict(params_b=pb, family=fam, file_gb=FILE_BYTES[model] / 1e9, gpu_pp=med(g, "pp"), cpu_pp=med(c, "pp"), gpu_tg=med(g, "tg"),
                       cpu_tg=med(c, "tg"), rounds=min(len(g), len(c)),
                       same_text=sorted({x["text"] for x in g}) == sorted({x["text"] for x in c}))
            dres["models"][model] = row
            rows.append((model, row))
            md.append(f"| {model} | {pb:.2f} | {row['gpu_pp']:.1f} | {row['cpu_pp']:.1f} | {row['gpu_tg']:.2f} | "
                      f"{row['cpu_tg']:.2f} | {row['gpu_tg'] / row['cpu_tg']:.2f} | {row['rounds']} | "
                      f"{'yes' if row['same_text'] else 'no'} |")
        md.append("")
        if len(rows) < 3:
            md += [f"(fewer than 3 models on {dev}: no fit)", ""]
            result["devices"][dev] = dres
            continue
        md += ["| subset | phase | side | inv-affine a (ms) | b (ms per x) | best law (rms rel) | inv-affine rms / LOO | "
               "power k | errors of all laws (rms rel) |", "|---|---|---|---|---|---|---|---|---|"]
        cross_lines = []
        for si, (label, xf, keep) in enumerate(SUBSETS):
            sub = [r for _, r in rows if keep(r)]
            if len(sub) < 3:
                md.append(f"| {label} | - | - | fewer than 3 models | | | | | |")
                continue
            P = [xf(r) for r in sub]
            sfits, scross = {}, {}
            for phase in ("pp", "tg"):
                fits = {}
                for side in sides:
                    v = [r[f"{side}_{phase}"] for r in sub]
                    f = fit_all(P, v)
                    fits[side] = f
                    best = min(f, key=lambda k: f[k]["rms_rel"])
                    ia = f["inv-affine"]
                    loo = f"{ia['loo_rms_rel'] * 100:.1f} %" if ia["loo_rms_rel"] is not None else "-"
                    alll = " ".join(f"{k} {f[k]['rms_rel'] * 100:.0f}" for k in f)
                    md.append(f"| {label} | {'prompt' if phase == 'pp' else 'generation'} | {side} | "
                              f"{ia['coef'][0] * 1e3:.2f} | {ia['coef'][1] * 1e3:.2f} | {best} "
                              f"({f[best]['rms_rel'] * 100:.1f} %) | {ia['rms_rel'] * 100:.1f} % / {loo} | "
                              f"{f['power']['coef'][1]:.2f} | {alll} |")
                sfits[phase] = fits
                if not cpu_ok:
                    continue
                ag, bg = fits["gpu"]["inv-affine"]["coef"]
                ac, bc = fits["cpu"]["inv-affine"]["coef"]
                scross[phase] = (ag - ac) / (bc - bg) if bc != bg else math.inf
            dres.setdefault("subsets", {})[label] = dict(fits=sfits, crossover=scross)
            if cpu_ok:
                cross_lines.append(f"{label}: " + ", ".join(
                    f"{'prompt' if k == 'pp' else 'generation'} x* = {v:.2f}" for k, v in scross.items()))
            if si == 0:  # the charts use the first subset (all models against parameters)
                dres["fits"], dres["crossover_b"] = sfits, scross
        md += ["", "Crossover x* where GPU and CPU take the same time (inv-affine; negative = one side faster at "
               "every size):", ""] + [f"- {c}" for c in cross_lines] + [""]
        result["devices"][dev] = dres
        plot(dev, rows, dres, out)
        md += [f"![{dev} ms per token](ladder_{dev}_ms.png)", f"![{dev} t/s](ladder_{dev}_tps.png)", ""]
    plot_fleet(result, out)
    md += ["![GPU/CPU ratio by size](ladder_fleet_ratio.png)", ""]
    (out / "ladder_fit.json").write_text(json.dumps(result, indent=1))
    (out / "ladder_fit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def plot_fleet(result, out):
    """GPU/CPU speed ratio against parameters, one line per device, prompt and generation side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, phase, title in ((axes[0], "pp", "prompt"), (axes[1], "tg", "generation")):
        for dev, d in sorted(result["devices"].items()):
            if not d["cpu_reference"]:
                continue
            pts = sorted((r["params_b"], r[f"gpu_{phase}"] / r[f"cpu_{phase}"]) for r in d["models"].values())
            if not pts:
                continue
            cores = ",".join(d["cpus"])
            ax.plot([p for p, _ in pts], [q for _, q in pts], marker="o", label=f"{dev} (CPU cores {cores})")
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_title(f"GPU / CPU speed, {title} (above 1: GPU faster)")
        ax.set_xlabel("parameters (B)")
        ax.set_ylabel("GPU t/s / CPU t/s (log)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ladder_fleet_ratio.png", dpi=120)
    plt.close(fig)


def plot(dev, rows, dres, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"gpu": "#1f6fb2", "cpu": "#c0504d"}
    mk = {"Qwen2.5": "o", "Llama 3.2": "s", "Gemma 2": "^"}
    xs = np.linspace(0.3, 3.5, 100)
    for kind in ("ms", "tps"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, phase, title in ((axes[0], "pp", "prompt (~370 tokens)"), (axes[1], "tg", "generation (64 tokens)")):
            for side in dres["fits"][phase]:
                for _, r in rows:
                    v = r[f"{side}_{phase}"]
                    y = 1e3 / v if kind == "ms" else v
                    ax.scatter(r["params_b"], y, color=col[side], marker=mk[r["family"]], zorder=3)
                a, b = dres["fits"][phase][side]["inv-affine"]["coef"]
                y = (a + b * xs) * 1e3 if kind == "ms" else 1.0 / (a + b * xs)
                ax.plot(xs, y, color=col[side], lw=1.5, label=f"{side.upper()}: {a * 1e3:.1f} + {b * 1e3:.1f}*P ms")
            ps = dres["crossover_b"].get(phase, math.inf)
            if 0.3 < ps < 3.5:
                ax.axvline(ps, color="gray", ls="--", lw=1)
                ax.text(ps, ax.get_ylim()[1] * 0.95, f" P*={ps:.2f}B", color="gray", va="top")
            ax.set_title(f"{dev} {title}")
            ax.set_xlabel("parameters (B)")
            ax.set_ylabel("ms per token" if kind == "ms" else "tokens per second")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.text(0.01, 0.01, "markers: circle Qwen2.5, square Llama 3.2, triangle Gemma 2; lines: inv-affine fit",
                 fontsize=7, color="gray")
        fig.tight_layout()
        fig.savefig(out / f"ladder_{dev}_{kind}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
