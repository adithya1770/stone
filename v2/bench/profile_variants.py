"""Measure per-variant inference latency on THIS device.

    python -m v2.bench.profile_variants --condition idle
    python -m v2.bench.profile_variants --condition stressed

Runs are interleaved across variants (round-robin, shuffled each round) so
thermal drift affects every variant equally. Writes one row per run
(raw samples, used by replay) plus a JSON summary.
"""
import argparse
import csv
import json
import os
import platform
import random
import statistics
import time

import numpy as np

from v2.common.paths import assert_inside_v2, enter_repo, results_path
from v2.common.engine import MultiVariantEngine, default_variants, load_rgb
from v2.common.datasets import get_source
from v2.routing.telemetry import BackgroundTelemetry
from v2.bench.stress import Stress


def device_info():
    model = platform.machine()
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            with open(p) as f:
                model = f.read().strip("\x00\n ")
                break
        except OSError:
            pass
    return {"machine": platform.machine(), "model": model, "cpus": os.cpu_count(),
            "python": platform.python_version()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["idle", "stressed"], default="idle")
    ap.add_argument("--variants", nargs="+", default=None, help="default: built-in ladder + registered variants")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dataset", default="auto", help="auto|imagenette|tiny|his_pool|<dir>")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--stressor", default="auto", choices=["auto", "stress-ng", "python"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="idle seconds before each timed run (0.5 mimics the original protocol, "
                         "whose blocking telemetry read leaves the CPU idle between requests)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    enter_repo()
    a.variants = a.variants or default_variants()
    ds = a.dataset
    if ds == "auto":
        ds = "imagenette" if os.path.isdir("datasets/imagenette2-320/val") else "tiny"
    items = get_source(ds, limit=a.images, seed=1)
    imgs = [load_rgb(p) for _, p, _ in items]

    eng = MultiVariantEngine(a.variants, num_threads=a.threads)
    tel = BackgroundTelemetry().start()
    rss = tel.rss_mb()
    tensors = {v: [eng.preprocess(im, v) for im in imgs] for v in a.variants}

    tag = f"_{a.tag}" if a.tag else ""
    out = assert_inside_v2(a.out or results_path("profile", f"profile_{a.condition}{tag}.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stress = Stress(prefer=a.stressor).start() if a.condition == "stressed" else None
    rows = []
    try:
        for v in a.variants:  # warmup
            m = eng.get(v)
            for k in range(a.warmup):
                m["it"].set_tensor(m["inp"]["index"], tensors[v][k % len(imgs)])
                m["it"].invoke()
        t_start = tel.snapshot()
        order = list(a.variants)
        rng = random.Random(0)
        for r in range(a.runs):
            rng.shuffle(order)
            for v in order:
                m = eng.get(v)
                m["it"].set_tensor(m["inp"]["index"], tensors[v][r % len(imgs)])
                if a.gap:
                    time.sleep(a.gap)
                t0 = time.perf_counter()
                m["it"].invoke()
                ms = (time.perf_counter() - t0) * 1000
                s = tel.snapshot()
                rows.append({"condition": a.condition, "variant": v, "run": r,
                             "latency_ms": round(ms, 3), "temperature": s["temperature"],
                             "cpu_external": s["cpu_external"], "throttled": s["throttled"]})
        t_end = tel.snapshot()
    finally:
        if stress:
            stress.stop()

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"device": device_info(), "condition": a.condition, "dataset": ds, "gap_s": a.gap,
               "stressor": stress.kind if stress else None, "runs": a.runs,
               "threads": a.threads, "rss_mb_all_variants": round(rss, 1),
               "temp_start": t_start["temperature"], "temp_end": t_end["temperature"],
               "throttled_end": t_end["throttled"], "variants": {}}
    print(f"\n{a.condition.upper()} on {summary['device']['model']} "
          f"({summary['device']['cpus']} cpus, stressor={summary['stressor']})")
    print(f"{'variant':<12}{'median':>9}{'mean':>9}{'p95':>9}  (ms)")
    for v in a.variants:
        xs = np.array([r["latency_ms"] for r in rows if r["variant"] == v])
        d = {"median": round(float(np.median(xs)), 3), "mean": round(float(xs.mean()), 3),
             "p95": round(float(np.percentile(xs, 95)), 3), "n": int(len(xs))}
        summary["variants"][v] = d
        print(f"{v:<12}{d['median']:>9.2f}{d['mean']:>9.2f}{d['p95']:>9.2f}")
    with open(out.replace(".csv", ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
