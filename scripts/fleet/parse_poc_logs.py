#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parse AMUDA CUDA-like POC fleet logs into one table.

    python3 scripts/fleet/parse_poc_logs.py evidence/poc_cuda_like/<timestamp>
    python3 scripts/fleet/parse_poc_logs.py --latest evidence/poc_cuda_like

Reads <host>.log written by run_poc_fleet.ps1 and prints a Markdown report:
per host/backend/sample status, error metric, GPU time, and the first error detail
(shader compiler file:line, runtime error name/detail, or connection failure).
Hosts that produced no AMU_RESULT lines are reported explicitly; nothing is dropped.
"""
import os
import re
import sys

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
HOSTS_ORDER = ["s25", "s21", "s20", "a35", "a53"]


def kv(line):
    return {k: v.strip('"') for k, v in KV.findall(line)}


def read_text(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_host(path):
    text = read_text(path)
    h = {"device": "", "soc": "", "android": "", "gpu": {}, "results": [], "summary": [], "fatal": [],
         "shader_errors": {}, "runlog": {}, "crash": {}, "exit": None, "unreachable": None}
    cur_amudacc = None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("UNREACHABLE"):
            h["unreachable"] = s
        m2 = re.match(r"\[amuda-poc\] (wake_lock=\S+|opencl LD_LIBRARY_PATH=.*)$", s)
        if m2:
            h.setdefault("env", []).append(m2.group(1))
        m = re.match(r"\[amuda-poc\] model=(.*?) soc=(.*?) android=(.*)$", s)
        if m:
            h["device"], h["soc"], h["android"] = m.groups()
        if s.startswith("AMU_DEVICE"):
            d = kv(s)
            mm = re.search(r'device="([^"]*)"', s)
            h["gpu"][d.get("backend", "?")] = mm.group(1) if mm else "(init failed)"
        elif s.startswith("AMU_RESULT"):
            r = kv(s)
            h["results"].append(r)
            if r.get("status") == "BUILD_ERROR":
                cur_amudacc = r.get("sample")
        elif s.startswith("AMU_SUMMARY"):
            h["summary"].append(kv(s))
            if "build" in kv(s):
                h["fatal"].append(s)
        elif s.startswith("AMUDACC|") or s.startswith("CC|"):
            body = s.split("|", 1)[1].strip()
            if cur_amudacc:
                h["shader_errors"].setdefault(cur_amudacc, []).append(body)
        elif s.startswith("BUILDLOG|"):
            h["fatal"].append(s.split("|", 1)[1].strip())
        elif s.startswith("RUNLOG|"):
            parts = s.split("|", 3)
            if len(parts) == 4:
                h["runlog"].setdefault((parts[1], parts[2]), []).append(parts[3].strip())
                # a sample that failed before naming itself prints sample=?; RUNLOG carries the real name
                if h["results"] and h["results"][-1].get("sample") == "?" and \
                        h["results"][-1].get("backend") == parts[1]:
                    h["results"][-1]["sample"] = parts[2]
        elif s.startswith("CRASH|"):
            parts = s.split("|", 3)
            if len(parts) == 4:
                h["crash"].setdefault((parts[1], parts[2]), []).append(parts[3].strip())
        elif s.startswith("TRACE|"):
            parts = s.split("|", 3)
            if len(parts) == 4:
                h.setdefault("trace", []).append((parts[1], parts[2], parts[3].strip()))
        elif s.startswith("REMOTE_EXIT="):
            h["exit"] = s.split("=", 1)[1]
        elif "[amuda-poc] ERROR:" in s:
            h["fatal"].append(s)
    return h


def first_error(h, r):
    st = r.get("status")
    sample, backend = r.get("sample"), r.get("backend")
    if st == "BUILD_ERROR":
        lines = h["shader_errors"].get(sample, [])
        loc = [l for l in lines if re.search(r"\w+\.(acu|comp):\d+", l)]
        return (loc or lines or ["(no compiler output captured)"])[0][:160]
    if st in ("ERROR", "FAIL"):
        crash = h["crash"].get((backend, sample), [])
        if crash:
            # the fault line plus the first frame outside libc names the culprit
            sig = next((l for l in crash if "signal" in l), crash[0])
            frame = next((l for l in crash if " pc " in l and "/libc.so" not in l), "")
            return (re.sub(r"^.*?(signal \d+.*)$", r"\1", sig)[:90] + " @ " +
                    re.sub(r"^.*?#\d+ pc \S+\s+", "", frame)[:110]) if frame else sig[:200]
        rl = h["runlog"].get((backend, sample), [])
        for key in ("detail:", "mismatch", "bin ", "->"):
            det = [l for l in rl if key in l]
            if det:
                return det[0][:200]
        return r.get("error", "") or (rl[0][:160] if rl else "")
    return ""


def metric(r):
    for k in ("max_abs_err", "max_rel_err", "rel_err", "wrong_bins"):
        if k in r:
            return "%s=%s" % (k, r[k])
    return ""


def main(argv):
    if len(argv) >= 2 and argv[0] == "--latest":
        base = argv[1]
        runs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        if not runs:
            print("no runs under %s" % base)
            return 2
        run = os.path.join(base, runs[-1])
    elif argv:
        run = argv[0]
    else:
        print(__doc__)
        return 2
    logs = {os.path.splitext(f)[0]: os.path.join(run, f) for f in os.listdir(run) if f.endswith(".log")}
    hosts = [x for x in HOSTS_ORDER if x in logs] + sorted(x for x in logs if x not in HOSTS_ORDER)
    print("# AMUDA CUDA-like POC fleet report: %s\n" % os.path.basename(os.path.normpath(run)))
    missing = [x for x in HOSTS_ORDER if x not in logs]
    if missing:
        print("Hosts with no log file at all: %s\n" % ", ".join(missing))
    print("| host | device / SoC | backend | GPU | sample | status | error metric | gpu ms | wall ms | first error |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    totals = {}
    for host in hosts:
        h = parse_host(logs[host])
        dev = "%s / %s" % (h["device"] or "?", h["soc"] or "?")
        if h["unreachable"]:
            print("| %s | %s | - | - | - | UNREACHABLE | - | - | - | %s |" % (host, dev, h["unreachable"]))
            totals[host] = "UNREACHABLE"
            continue
        if not h["results"]:
            why = (h["fatal"][:1] or ["no AMU_RESULT lines (exit=%s)" % h["exit"]])[0]
            print("| %s | %s | - | - | - | NO_RESULTS | - | - | - | %s |" % (host, dev, why[:160]))
            totals[host] = "NO_RESULTS"
            continue
        npass = sum(1 for r in h["results"] if r.get("status") == "PASS")
        totals[host] = "%d PASS / %d total (exit=%s)" % (npass, len(h["results"]), h["exit"])
        for r in h["results"]:
            b = r.get("backend", "?")
            print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                host, dev, b, h["gpu"].get(b, "-"), r.get("sample", "?"), r.get("status", "?"), metric(r),
                r.get("gpu_ms", "-"), r.get("wall_ms", "-"), first_error(h, r).replace("|", "/")))
    print("\n## Per host\n")
    for host in hosts:
        env = parse_host(logs[host]).get("env", [])
        print("- %s: %s%s" % (host, totals.get(host), ("  [" + "; ".join(env) + "]") if env else ""))
    traced = [(host, parse_host(logs[host]).get("trace", [])) for host in hosts]
    if any(t for _, t in traced):
        print("\n## Launch phase trace (AMUDA_TRACE=1, vector_add: warm-up launch first, then 10 timed)\n")
        for host, tr in traced:
            for backend, sample, ln in tr:
                print("- %s %s %s: %s" % (host, backend, sample, ln))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
