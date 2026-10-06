"""Table-I style analysis of a v2 suite run, side by side with the original.

    python -m v2.analyse_v2 --suite v2/results_v2/suite/<tag>

Reports, per config (mean ± std over trials), on STRESSED rows (cpu >= 70,
the original definition):
  stressed latency, p95 latency, stressed accuracy (+ Wilson 95% CI on the
  stressed images), INT8 usage (original configs) / escalation rate (v2),
  end-to-end request time and process CPU time per request.
Plus:
  - paired comparison vs Baseline on the same stressed images (wins/losses,
    exact McNemar p-value)
  - the weighted-coin-flip accuracy between the re-run AlwaysINT8 and
    Baseline at each config's latency (the fair 'no intelligence' line)
  - the original paper's Table I numbers (results/stone_pi_multitrial2)
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics as stt

from v2.common.paths import assert_inside_v2, enter_repo

STRESS_CPU = 70.0
ORDER = ["baseline", "always_int8", "linucb_cold", "linucb_warm", "egreedy", "eightsignal_cold",
         "eightsignal_warm", "v2_cascade", "v2_resource_aware", "baseline_fixed", "v2_resource_aware_fixed",
         "v2_fixed_8signal", "always_int8_rebuilt", "baseline_rebuilt", "v2_rebuilt", "eightsignal_rebuilt", "v2_final",
         "fixed_ladder", "big_ladder", "eightsignal_ladder", "v2_ladder", "cpu_ladder", "lite1_ladder", "egreedy_ladder", "linucb_ladder", "v2_ladder_8sig", "v2_ladder_cpu"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - m), 100 * (c + m))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


INT8_KINDS = {"int8", "mnv2q", "mnv3sq", "efl0q", "efl1q", "efl2q", "efl3q", "efl4q"}  # all full-integer models


def is_int8(model):
    return model in INT8_KINDS or model.rsplit("_", 1)[0] in INT8_KINDS


def run_metrics(rows):
    s = [r for r in rows if float(r["cpu"]) >= STRESS_CPU]
    known = [r for r in s if r["correct"] in ("True", "False")]
    lat = [float(r["latency_ms"]) for r in s]
    out = {"n_stress": len(s),
           "lat": stt.mean(lat) if lat else float("nan"),
           "p95": sorted(lat)[max(0, int(math.ceil(0.95 * len(lat))) - 1)] if lat else float("nan"),
           "acc": 100 * sum(r["correct"] == "True" for r in known) / len(known) if known else float("nan"),
           "k": sum(r["correct"] == "True" for r in known), "n": len(known),
           "acc_all": 100 * sum(r["correct"] == "True" for r in rows) / len(rows)}
    m = [r["model"] for r in s]
    out["int8"] = 100 * sum(is_int8(x) for x in m) / len(m) if m else 0.0
    if s and "escalated" in s[0]:
        out["esc"] = 100 * sum(r["escalated"] == "True" for r in s) / len(s)
        out["e2e"] = stt.mean(float(r["e2e_ms"]) for r in s)
        out["cpu_ms"] = stt.mean(float(r["cpu_time_ms"]) for r in s)
        tiers = [r.get("tier", "") for r in s if r.get("tier")]
        out["tier_stressed_pct"] = 100 * sum(t != "idle" for t in tiers) / len(tiers) if tiers else None
    out["correct_by_img"] = {r["image_path"] if "image_path" in r else r["iteration"]: r["correct"] == "True" for r in known}
    return out


def majority(runs):
    """image -> correct in the majority of trials."""
    votes = {}
    for r in runs:
        for img, ok in r["correct_by_img"].items():
            votes.setdefault(img, []).append(ok)
    return {img: sum(v) * 2 >= len(v) for img, v in votes.items()}


def ms(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return (float("nan"), 0.0)
    return (stt.mean(vals), stt.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--dataset", default="imagenette")
    ap.add_argument("--his-dir", default="results/stone_pi_multitrial2")
    ap.add_argument("--ref", default="baseline", help="config for the paired W/L test (e.g. always_int8)")
    a = ap.parse_args()
    enter_repo()

    trials = sorted(glob.glob(os.path.join(a.suite, "trial*", a.dataset)))
    per = {}
    for td in trials:
        for f in sorted(glob.glob(os.path.join(td, "*_log.csv"))):
            name = os.path.basename(f)[:-8]
            per.setdefault(name, []).append(run_metrics(read(f)))
    names = [n for n in ORDER if n in per] + [n for n in per if n not in ORDER]

    # coin-flip line from re-run statics
    mix = None
    if "baseline" in per and "always_int8" in per:
        bl, bi = ms([m["lat"] for m in per["baseline"]])[0], ms([m["lat"] for m in per["always_int8"]])[0]
        ba, ia = ms([m["acc"] for m in per["baseline"]])[0], ms([m["acc"] for m in per["always_int8"]])[0]

        def mix(lat):
            if lat < bi:
                return None
            p = min(1.0, (lat - bi) / max(bl - bi, 1e-9))
            return (1 - p) * ia + p * ba

    lines = [f"# STONE v2 — Pi protocol results", "", f"Suite: `{a.suite}` — {len(trials)} trial(s); "
             f"stressed rows = cpu ≥ {STRESS_CPU:.0f}% (original definition).", "",
             "| config | stressed lat (ms) | p95 (ms) | stressed acc (%) | acc 95% CI | escalated % / final answer from INT8 % | e2e (ms) | CPU ms/req | coin-flip acc @ same lat | vs " + a.ref + " per image (W/L, p) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    table = {}
    for n in names:
        R = per[n]
        L, P, A = ms([m["lat"] for m in R]), ms([m["p95"] for m in R]), ms([m["acc"] for m in R])
        k, nn = R[0]["k"], R[0]["n"]
        ci = wilson(round(A[0] / 100 * nn), nn)
        usage = f"{ms([m['int8'] for m in R])[0]:.1f} int8"
        if n.startswith("v2"):   # v2: share escalated, and share whose final answer came from an INT8 model
            usage = f"{ms([m['esc'] for m in R])[0]:.1f} esc, " + usage
        e2e = ms([m.get("e2e") for m in R])[0] if "e2e" in R[0] else float("nan")
        cpu = ms([m.get("cpu_ms") for m in R])[0] if "cpu_ms" in R[0] else float("nan")
        mx = mix(L[0]) if mix else None
        wl = ""
        if a.ref in per and n != a.ref:
            # Paired test on UNIQUE images. Every trial uses the same images, so
            # summing wins/losses over trials would count each image several
            # times and overstate significance. Each config's per-image outcome
            # is its majority over trials.
            maj_ref, maj_x = majority(per[a.ref]), majority(R)
            W = sum(1 for img in maj_ref if img in maj_x and maj_x[img] and not maj_ref[img])
            Ls = sum(1 for img in maj_ref if img in maj_x and maj_ref[img] and not maj_x[img])
            wl = f"{W}/{Ls}, p={mcnemar_exact(W, Ls):.2f}"
        table[n] = {"lat": L, "p95": P, "acc": A, "ci": ci, "usage": usage, "e2e": e2e, "cpu_ms": cpu, "coin_flip": mx, "w_l": wl}
        lines.append(f"| {n} | {L[0]:.1f} ± {L[1]:.1f} | {P[0]:.1f} | {A[0]:.1f} ± {A[1]:.1f} | [{ci[0]:.0f}, {ci[1]:.0f}] | {usage} | "
                     f"{e2e:.1f} | {cpu:.1f} | {('%.1f' % mx) if mx is not None else 'faster than INT8'} | {wl} |")

    # original Table I for reference
    if os.path.isdir(a.his_dir):
        lines += ["", f"## Original results (`{a.his_dir}`, recomputed with the same code)", "",
                  "| config | stressed lat (ms) | stressed acc (%) | INT8 % |", "|---|---|---|---|"]
        for n in ORDER:
            fs = sorted(glob.glob(os.path.join(a.his_dir, "trial*", "imagenette", f"{n}_log.csv")))
            if not fs:
                continue
            R = [run_metrics(read(f)) for f in fs]
            L, A, I = ms([m["lat"] for m in R]), ms([m["acc"] for m in R]), ms([m["int8"] for m in R])
            lines.append(f"| {n} | {L[0]:.1f} ± {L[1]:.1f} | {A[0]:.1f} ± {A[1]:.1f} | {I[0]:.1f} ± {I[1]:.1f} |")

    lines += ["", "Notes:", "- Accuracy CIs use the number of stressed images (same images every trial), "
              "so they show real uncertainty; the ± across trials only shows run-to-run agreement.",
              "- 'coin-flip acc' = weighted coin flip between the re-run AlwaysINT8 and Baseline at the same latency; "
              "a router with no per-request intelligence lands on this line.",
              "- W/L = stressed images a config gets right that the reference gets wrong / the reverse (unique images, "
              "majority over trials); p = exact McNemar test.",
              "- e2e = decode + preprocess + inference(s) for one request; CPU ms/req = process CPU time per request."]
    out_dir = assert_inside_v2(os.path.join(a.suite, "analysis"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(table, f, indent=2, default=str)
    _plot(table, mix, out_dir)
    print("\n".join(lines))
    print(f"\n-> {out_dir}/results.md")


def _plot(table, mix, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for n, t in table.items():
        col = "#1a9850" if n.startswith("v2") else "#555"
        ax.errorbar(t["lat"][0], t["acc"][0], xerr=t["lat"][1],
                    yerr=[[t["acc"][0] - t["ci"][0]], [t["ci"][1] - t["acc"][0]]],
                    fmt="o", color=col, capsize=3, alpha=0.9)
        ax.annotate(n, (t["lat"][0], t["acc"][0]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    if "baseline" in table and "always_int8" in table:
        b, i = table["baseline"], table["always_int8"]
        ax.plot([i["lat"][0], b["lat"][0]], [i["acc"][0], b["acc"][0]], "--", color="#d9822b",
                label="coin flip INT8↔FP32 (no per-request intelligence)")
    ax.set_xlabel("stressed latency (ms)")
    ax.set_ylabel("stressed accuracy (%) with 95% CI")
    ax.set_title("Pi protocol: accuracy vs latency under stress")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_latency.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
