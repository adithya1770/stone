"""Run EVERY variant on EVERY image once -> the response table.

    python -m v2.bench.build_response_table --dataset imagenette --exclude-his-pool --limit 1000

One row per (image, variant): correct?, confidence, margin, entropy,
latency. With this table any routing policy can be evaluated offline,
exactly and instantly (bench/replay_eval.py), instead of re-running the
device for every policy.

--exclude-his-pool keeps the original 100-image evaluation sequence out of
the table, so thresholds tuned on it are never tested on the same images
in run_suite_v2.py (no data leakage).
"""
import argparse
import csv
import os
import time

from v2.common.paths import assert_inside_v2, enter_repo, results_path
from v2.common.engine import MultiVariantEngine, default_variants, load_rgb
from v2.common.datasets import get_source

FIELDS = ["image_id", "true_index", "variant", "pred_index", "correct",
          "confidence", "margin", "entropy", "latency_ms", "preprocess_ms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="auto", help="auto|imagenette|tiny|his_pool|<dir>")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-his-pool", action="store_true")
    ap.add_argument("--variants", nargs="+", default=None, help="default: built-in ladder + registered variants")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true", help="append to --out, skipping images already done")
    ap.add_argument("--max-seconds", type=float, default=None, help="stop after this long (use with --resume)")
    a = ap.parse_args()

    enter_repo()
    a.variants = a.variants or default_variants()
    ds = a.dataset
    if ds == "auto":
        ds = "imagenette" if os.path.isdir("datasets/imagenette2-320/val") else "tiny"
    items = get_source(ds, limit=a.limit, seed=a.seed, exclude_his_pool=a.exclude_his_pool)
    items = [it for it in items if it[2] is not None]
    out = assert_inside_v2(a.out or results_path("tables", f"table_{ds}.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    eng = MultiVariantEngine(a.variants, num_threads=a.threads)

    t0 = time.time()
    n_ok = {v: 0 for v in a.variants}
    done = set()
    if a.resume and os.path.exists(out):
        with open(out) as f:
            seen = {}
            for r in csv.DictReader(f):
                seen.setdefault(r["image_id"], set()).add(r["variant"])
        done = {i for i, vs in seen.items() if set(a.variants) <= vs}
    todo = [it for it in items if it[0] not in done]
    print(f"{len(done)} images already done, {len(todo)} to go")
    append = a.resume and os.path.exists(out)
    with open(out, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not append:
            w.writeheader()
        for i, (iid, path, true_idx) in enumerate(todo):
            if a.max_seconds and time.time() - t0 > a.max_seconds:
                print(f"time limit: stopped after {i} images ({len(todo) - i} left)")
                break
            img = load_rgb(path)
            for v in a.variants:
                r = eng.run_image(img, v)
                ok = r["index"] == true_idx
                n_ok[v] += ok
                w.writerow({"image_id": iid, "true_index": true_idx, "variant": v,
                            "pred_index": r["index"], "correct": ok,
                            "confidence": r["confidence"], "margin": r["margin"],
                            "entropy": r["entropy"], "latency_ms": r["latency_ms"],
                            "preprocess_ms": r["preprocess_ms"]})
            if (i + 1) % 100 == 0 or i + 1 == len(todo):
                accs = " ".join(f"{v}={100*n_ok[v]/(i+1):.1f}" for v in a.variants)
                print(f"[{i+1}/{len(todo)}] {time.time()-t0:.0f}s  acc%: {accs}", flush=True)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
