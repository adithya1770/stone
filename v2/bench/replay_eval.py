"""Offline replay: evaluate routing policies exactly, from the response table.

    python -m v2.bench.replay_eval \
        --table v2/results_v2/tables/table_imagenette.csv \
        --profile-idle v2/results_v2/profile/profile_idle.csv \
        --profile-stressed v2/results_v2/profile/profile_stressed.csv \
        --write-config

For each device condition (idle / stressed):
  1. latency of each variant is drawn from that condition's measured
     profile samples (common random numbers: every policy sees the same
     latency draw for the same image+variant)
  2. images are split deterministically into TUNE and TEST halves
  3. on TUNE: pick the best (first model, escalation model, margin
     threshold) under the latency budget (default: AlwaysINT8's latency)
  4. on TEST (never seen in tuning): report accuracy, mean/p95 latency,
     escalation rate, paired-bootstrap CIs vs Baseline and AlwaysINT8,
     the weighted-coin-flip line of the original FP32/INT8 design space,
     and the per-image oracle
  5. run the live ResourceAwareCascade class (with online budget
     adaptation) over TEST, and over a mixed idle->stressed schedule
Writes report.md, summary.json, Pareto plots and (optionally)
v2/configs/operating_points.json used by the runtime.
"""
import argparse
import csv
import datetime
import json
import os

import numpy as np

from v2.common.paths import CONFIG_ROOT, RESULTS_ROOT, assert_inside_v2, enter_repo
from v2.common.datasets import split_of
from v2.routing.policies import ResourceAwareCascade
from v2.routing.budget import SIGNALS

BASE, INT8 = "fp32_224", "int8_224"


# ---------------------------------------------------------------- loading
def load_table(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    variants, ids, seen = [], [], set()
    by = {}
    for r in rows:
        v, i = r["variant"], r["image_id"]
        if v not in variants:
            variants.append(v)
        if i not in seen:
            seen.add(i)
            ids.append(i)
        by[(i, v)] = r
    ids = [i for i in ids if all((i, v) in by for v in variants)]
    T = {"ids": ids, "variants": variants, "correct": {}, "margin": {}, "conf": {}, "ent": {}, "lat": {}}
    for v in variants:
        T["correct"][v] = np.array([by[(i, v)]["correct"] == "True" for i in ids])
        T["margin"][v] = np.array([float(by[(i, v)]["margin"]) for i in ids])
        T["conf"][v] = np.array([float(by[(i, v)]["confidence"]) for i in ids])
        T["ent"][v] = np.array([float(by[(i, v)]["entropy"]) for i in ids])
        T["lat"][v] = np.array([float(by[(i, v)]["latency_ms"]) for i in ids])
    return T


def load_profile(path):
    if not path or not os.path.exists(path):
        return None
    s = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            s.setdefault(r["variant"], []).append(float(r["latency_ms"]))
    return {v: np.array(x) for v, x in s.items()}


def latency_matrix(T, profile, seed):
    """Per (variant, image) latency draw from the condition's profile."""
    rng = np.random.default_rng(seed)
    n = len(T["ids"])
    L, src = {}, {}
    for v in T["variants"]:
        if profile is not None and v in profile and len(profile[v]):
            L[v] = rng.choice(profile[v], size=n, replace=True)
            src[v] = "profile"
        else:
            L[v] = T["lat"][v].copy()
            src[v] = "table"
    return L, src


# ---------------------------------------------------------------- metrics
def _summ(correct, lat):
    return {"acc": float(correct.mean() * 100), "lat": float(lat.mean()),
            "p95": float(np.percentile(lat, 95)), "n": int(len(correct))}


def static_eval(T, L, v, idx):
    m = _summ(T["correct"][v][idx], L[v][idx])
    m.update(kind="static", first=v, second=None, theta=None, esc=0.0)
    return m


def score(T, signal, v):
    """Higher = more confident, matching routing.budget.signal_score."""
    return {"margin": T["margin"][v], "confidence": T["conf"][v], "entropy": -T["ent"][v]}[signal]


def cascade_arrays(T, L, a, b, theta, idx, signal="margin"):
    esc = score(T, signal, a)[idx] < theta
    corr = np.where(esc, T["correct"][b][idx], T["correct"][a][idx])
    lat = L[a][idx] + esc * L[b][idx]
    return corr, lat, esc


def cascade_eval(T, L, a, b, theta, idx, signal="margin"):
    corr, lat, esc = cascade_arrays(T, L, a, b, theta, idx, signal)
    m = _summ(corr, lat)
    m.update(kind="cascade", first=a, second=b, theta=float(theta), signal=signal, esc=float(esc.mean() * 100))
    return m


def candidates(T, L, idx, n_theta=41, signals=tuple(SIGNALS)):
    V = T.get("cands", T["variants"])   # variants v2 may use (references excluded)
    mean_lat = {v: L[v][idx].mean() for v in V}
    out = [static_eval(T, L, v, idx) for v in V]
    for sig in signals:
        lo, hi = SIGNALS[sig]
        for a in V:
            sc = score(T, sig, a)[idx]
            qs = np.unique(np.concatenate([[lo], np.quantile(sc, np.linspace(0, 1, n_theta)), [hi + 0.01]]))
            for b in V:
                if b == a or mean_lat[b] <= mean_lat[a]:
                    continue
                for th in qs:
                    out.append(cascade_eval(T, L, a, b, th, idx, sig))
    return out


def select(cands, budget, acc_tol=0.0):
    """Max accuracy with mean latency <= budget. acc_tol (percentage points):
    among configs within acc_tol of that best accuracy, take the fastest, so a
    difference of one or two tuning images does not buy a much slower config."""
    feas = [c for c in cands if c["lat"] <= budget]
    if not feas:
        return min(cands, key=lambda c: c["lat"])
    best = max(c["acc"] for c in feas)
    return min([c for c in feas if c["acc"] >= best - acc_tol - 1e-9], key=lambda c: c["lat"])


def select_min_latency(cands, acc_target):
    feas = [c for c in cands if c["acc"] >= acc_target - 1e-9]
    return min(feas, key=lambda c: c["lat"]) if feas else None


def pareto(points):
    pts = sorted(points, key=lambda c: (c["lat"], -c["acc"]))
    front, best = [], -1
    for c in pts:
        if c["acc"] > best + 1e-9:
            front.append(c)
            best = c["acc"]
    return front


def config_arrays(T, L, cfg, idx):
    if cfg["second"] is None:
        return T["correct"][cfg["first"]][idx], L[cfg["first"]][idx]
    corr, lat, _ = cascade_arrays(T, L, cfg["first"], cfg["second"], cfg["theta"], idx, cfg.get("signal", "margin"))
    return corr, lat


def paired_bootstrap(T, L, cfg, ref, idx, B=2000, seed=0):
    c1, l1 = config_arrays(T, L, cfg, idx)
    c2, l2 = T["correct"][ref][idx], L[ref][idx]
    dc = c1.astype(float) - c2.astype(float)
    dl = l1 - l2
    rng = np.random.default_rng(seed)
    n = len(idx)
    bs_c, bs_l = [], []
    for _ in range(B):
        s = rng.integers(0, n, n)
        bs_c.append(dc[s].mean() * 100)
        bs_l.append(dl[s].mean())
    return {"d_acc": float(dc.mean() * 100), "d_acc_ci": [float(np.percentile(bs_c, 2.5)), float(np.percentile(bs_c, 97.5))],
            "d_lat": float(dl.mean()), "d_lat_ci": [float(np.percentile(bs_l, 2.5)), float(np.percentile(bs_l, 97.5))],
            "wins": int(((c1 == 1) & (c2 == 0)).sum()), "losses": int(((c1 == 0) & (c2 == 1)).sum())}


def mix_line_acc(st, lat):
    """Accuracy of the weighted coin flip between AlwaysINT8 and Baseline
    (the original design space) at the given mean latency."""
    a, b = st[INT8], st[BASE]
    if lat < a["lat"]:
        return None
    p = min(1.0, (lat - a["lat"]) / max(b["lat"] - a["lat"], 1e-9))
    return (1 - p) * a["acc"] + p * b["acc"]


def auroc(score, label):
    """P(score of a correct prediction > score of a wrong one); 0.5 = useless.
    Measures how well a signal (margin, confidence) flags wrong answers."""
    label = np.asarray(label, bool)
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sv = np.asarray(score)[order]
    i = 0
    while i < len(sv):  # average ranks for ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def oracle_eval(T, L, idx):
    order = sorted(T.get("cands", T["variants"]), key=lambda v: L[v][idx].mean())
    corr = np.zeros(len(idx), bool)
    lat = np.zeros(len(idx))
    for k, i in enumerate(idx):
        pick = next((v for v in order if T["correct"][v][i]), order[0])
        corr[k] = T["correct"][pick][i]
        lat[k] = L[pick][i]
    return _summ(corr, lat)


def simulate_policy(T, Ls, policy, idx, tiers):
    """Run a live Policy object over images in order; tiers[k] in Ls keys."""
    corr = np.zeros(len(idx), bool)
    lat = np.zeros(len(idx))
    esc = np.zeros(len(idx), bool)
    thetas = []
    for k, i in enumerate(idx):
        tier = tiers[k]
        L = Ls[tier]
        ctx = {"tier": tier}
        v1 = policy.first(ctx)
        r1 = {"variant": v1, "margin": float(T["margin"][v1][i]), "confidence": float(T["conf"][v1][i]),
              "entropy": float(T["ent"][v1][i]), "latency_ms": float(L[v1][i])}
        v2 = policy.second(ctx, r1)
        r2 = None
        total = r1["latency_ms"]
        final = v1
        if v2:
            r2 = {"variant": v2, "margin": float(T["margin"][v2][i]), "confidence": float(T["conf"][v2][i]),
                  "entropy": float(T["ent"][v2][i]), "latency_ms": float(L[v2][i])}
            total += r2["latency_ms"]
            final = v2
            esc[k] = True
        policy.observe(ctx, [r1, r2], total)
        corr[k] = T["correct"][final][i]
        lat[k] = total
        thetas.append(policy.state())
    m = _summ(corr, lat)
    m["esc"] = float(esc.mean() * 100)
    m["final_theta"] = thetas[-1] if thetas else {}
    return m, corr, lat


# ---------------------------------------------------------------- plotting
def plot(cond, test_cands, st, chosen, oracle, budget, out_png, title_extra=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    front = pareto([c for c in test_cands if c["kind"] == "cascade"] + [c for c in test_cands if c["kind"] == "static"])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot([c["lat"] for c in front], [c["acc"] for c in front], "-", color="#2a6fdb", lw=2,
            label="v2 cascade frontier (test)")
    for v, s in st.items():
        ax.scatter(s["lat"], s["acc"], color="#888", s=28, zorder=3)
        ax.annotate(v, (s["lat"], s["acc"]), fontsize=7, xytext=(3, -9), textcoords="offset points", color="#555")
    a, b = st[INT8], st[BASE]
    ax.plot([a["lat"], b["lat"]], [a["acc"], b["acc"]], "--", color="#d9822b", lw=1.6,
            label="original design space (coin flip INT8<->FP32)")
    ax.scatter(chosen["lat"], chosen["acc"], marker="*", s=260, color="#1a9850", zorder=5,
               label=f"chosen v2 operating point")
    ax.scatter(oracle["lat"], oracle["acc"], marker="^", s=70, color="#7b3294", zorder=4, label="per-image oracle")
    ax.axvline(budget, color="#1a9850", ls=":", lw=1, label=f"budget {budget:.1f} ms")
    ax.set_xlabel("mean latency per request (ms, invoke only)")
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title(f"{cond} — accuracy vs latency {title_extra}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--profile-idle", default=None)
    ap.add_argument("--profile-stressed", default=None)
    ap.add_argument("--budget-idle", type=float, default=None, help="default: AlwaysINT8 mean latency (idle)")
    ap.add_argument("--budget-stressed", type=float, default=None, help="default: AlwaysINT8 mean latency (stressed)")
    ap.add_argument("--budget-ref-idle", default=None,
                    help="budget = this variant's mean latency when idle (e.g. mnv2f_224: spare time may be spent on FP32)")
    ap.add_argument("--budget-ref-stressed", default=None,
                    help="budget = this variant's mean latency when stressed (e.g. mnv2q_224: stay at INT8 cost)")
    ap.add_argument("--budget-tol", type=float, default=0.0,
                    help="relative slack on the budget (0.05 = 5%%) so measurement noise cannot exclude the reference itself")
    ap.add_argument("--acc-tol", type=float, default=0.0,
                    help="pp: prefer the fastest config within this much of the best tuning accuracy")
    ap.add_argument("--tune-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument("--config-path", default=os.path.join(CONFIG_ROOT, "operating_points.json"))
    ap.add_argument("--label", default="")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="restrict to these variants (e.g. original-preprocessing only)")
    ap.add_argument("--original-only", action="store_true",
                    help="only the original models with the original preprocessing (kinds int8, fp32)")
    ap.add_argument("--kinds", nargs="+", default=None,
                    help="only these model kinds, e.g. int8 fp32 fp32n (fixed) or add mnv2q mnv2f (rebuilt)")
    a = ap.parse_args()
    enter_repo()

    T = load_table(a.table)
    keep = a.variants or T["variants"]
    kinds = ["int8", "fp32"] if a.original_only else a.kinds
    if kinds:
        keep = [v for v in keep if v.rsplit("_", 1)[0] in kinds]
    # v2 chooses only among `keep`; Baseline and AlwaysINT8 stay in the table
    # as reference rows (budget = AlwaysINT8 latency, paired tests vs both)
    T["cands"] = [v for v in T["variants"] if v in keep]
    T["variants"] = [v for v in T["variants"] if v in keep or v in (BASE, INT8)]
    ids = T["ids"]
    tune = np.array([k for k, i in enumerate(ids) if split_of(i, a.tune_frac) == "tune"])
    test = np.array([k for k, i in enumerate(ids) if split_of(i, a.tune_frac) == "test"])
    out_dir = a.out_dir or os.path.join(RESULTS_ROOT, "replay", os.path.splitext(os.path.basename(a.table))[0])
    os.makedirs(assert_inside_v2(out_dir), exist_ok=True)

    profiles = {"idle": load_profile(a.profile_idle), "stressed": load_profile(a.profile_stressed)}
    conds = [c for c in ("idle", "stressed") if profiles[c] is not None] or ["idle"]

    summary = {"table": a.table, "n_images": len(ids), "n_tune": len(tune), "n_test": len(test),
               "variants": T["variants"], "v2_candidates": T["cands"], "conditions": {}, "generated": datetime.datetime.now().isoformat()}
    Ls, op_points = {}, {}
    md = [f"# STONE v2 replay report {a.label}", "",
          f"Table: `{a.table}` — {len(ids)} images ({len(tune)} tune / {len(test)} test, split by hash).", ""]

    md += ["## How well does the cheap model's score flag its own mistakes? (AUROC, test split)", "",
           "0.5 = no better than chance, 1.0 = perfect. High AUROC is what makes escalation pay off.", "",
           "| variant | accuracy % | AUROC margin | AUROC confidence | AUROC -entropy |", "|---|---|---|---|---|"]
    summary["auroc"] = {}
    for v in T["variants"]:
        am = auroc(T["margin"][v][test], T["correct"][v][test])
        ac = auroc(T["conf"][v][test], T["correct"][v][test])
        ae = auroc(-T["ent"][v][test], T["correct"][v][test])
        summary["auroc"][v] = {"margin": am, "confidence": ac, "entropy": ae}
        md.append(f"| {v} | {100 * T['correct'][v][test].mean():.1f} | {am:.3f} | {ac:.3f} | {ae:.3f} |")
    md.append("")

    for ci, cond in enumerate(conds):
        L, src = latency_matrix(T, profiles[cond], seed=a.seed + ci)
        Ls[cond] = L
        st_tune = {v: static_eval(T, L, v, tune) for v in T["variants"]}
        st_test = {v: static_eval(T, L, v, test) for v in T["variants"]}
        ref = getattr(a, f"budget_ref_{cond}") or INT8
        if ref not in st_tune:
            raise SystemExit(f"--budget-ref-{cond} {ref}: not in the table")
        budget = (getattr(a, f"budget_{cond}") or st_tune[ref]["lat"]) * (1 + a.budget_tol)
        cands_tune = candidates(T, L, tune)
        chosen = select(cands_tune, budget, a.acc_tol)
        acc_match = select_min_latency(cands_tune, st_tune[BASE]["acc"])
        chosen_test = (static_eval(T, L, chosen["first"], test) if chosen["second"] is None
                       else cascade_eval(T, L, chosen["first"], chosen["second"], chosen["theta"], test, chosen["signal"]))
        acc_match_test = None
        if acc_match:
            acc_match_test = (static_eval(T, L, acc_match["first"], test) if acc_match["second"] is None
                              else cascade_eval(T, L, acc_match["first"], acc_match["second"], acc_match["theta"], test, acc_match["signal"]))
        cands_test = candidates(T, L, test)
        oracle = oracle_eval(T, L, test)
        vs_base = paired_bootstrap(T, L, chosen, BASE, test)
        vs_int8 = paired_bootstrap(T, L, chosen, INT8, test)
        mix = mix_line_acc(st_test, chosen_test["lat"])

        est = {v: round(float(L[v].mean()), 3) for v in T["variants"]}
        op = {"first": chosen["first"], "second": chosen["second"], "theta": chosen["theta"] if chosen["theta"] is not None else 0.0,
              "signal": chosen.get("signal") or "margin", "budget_ms": round(float(budget), 3), "budget_ref": ref, "est_ms": est}
        op_points[cond] = op

        # live policy class with online budget adaptation, TEST order
        pol = ResourceAwareCascade({cond: op})
        live, _, _ = simulate_policy(T, {cond: L}, pol, test, [cond] * len(test))

        png = plot(cond, cands_test, st_test, chosen_test, oracle, budget,
                   os.path.join(out_dir, f"pareto_{cond}.png"), a.label)
        summary["conditions"][cond] = {
            "latency_source": src, "budget_ms": budget, "static_test": st_test,
            "chosen_on_tune": chosen, "chosen_test": chosen_test,
            "accuracy_matched_on_tune": acc_match, "accuracy_matched_test": acc_match_test,
            "vs_baseline": vs_base, "vs_alwaysint8": vs_int8, "coin_flip_acc_at_same_latency": mix,
            "oracle_test": oracle, "live_policy_test": live, "plot": png}

        # markdown
        md += [f"## Condition: {cond}", "",
               f"Latency source: {sorted(set(src.values()))}; budget = {budget:.2f} ms "
               f"({('mean latency of ' + ref) if getattr(a, f'budget_{cond}') is None else 'user'}"
               f"{f' + {100 * a.budget_tol:.0f}% tolerance' if a.budget_tol else ''}"
               f"{f'; fastest config within {a.acc_tol:.1f} pp of the best' if a.acc_tol else ''})", "",
               "| config | acc % (test) | mean ms | p95 ms | escalated % |", "|---|---|---|---|---|"]
        for v in T["variants"]:
            s = st_test[v]
            md.append(f"| static {v} | {s['acc']:.1f} | {s['lat']:.2f} | {s['p95']:.2f} | – |")
        c = chosen_test
        name = c["first"] if c["second"] is None else f"{c['first']} → {c['second']} ({c['signal']}<{c['theta']:.3f})"
        md.append(f"| **v2 chosen: {name}** | **{c['acc']:.1f}** | **{c['lat']:.2f}** | {c['p95']:.2f} | {c['esc']:.1f} |")
        md.append(f"| v2 live policy (online budget) | {live['acc']:.1f} | {live['lat']:.2f} | {live['p95']:.2f} | {live['esc']:.1f} |")
        if acc_match_test:
            c = acc_match_test
            name = c["first"] if c["second"] is None else f"{c['first']} → {c['second']} ({c['signal']}<{c['theta']:.3f})"
            md.append(f"| v2 accuracy-matched: {name} | {c['acc']:.1f} | {c['lat']:.2f} | {c['p95']:.2f} | {c['esc']:.1f} |")
        md.append(f"| per-image oracle (upper bound) | {oracle['acc']:.1f} | {oracle['lat']:.2f} | {oracle['p95']:.2f} | – |")
        md += ["",
               f"- vs Baseline (fp32_224): Δacc {vs_base['d_acc']:+.1f} pp "
               f"[95% CI {vs_base['d_acc_ci'][0]:+.1f}, {vs_base['d_acc_ci'][1]:+.1f}], "
               f"Δlatency {vs_base['d_lat']:+.2f} ms [{vs_base['d_lat_ci'][0]:+.2f}, {vs_base['d_lat_ci'][1]:+.2f}]",
               f"- vs AlwaysINT8 (int8_224): Δacc {vs_int8['d_acc']:+.1f} pp "
               f"[95% CI {vs_int8['d_acc_ci'][0]:+.1f}, {vs_int8['d_acc_ci'][1]:+.1f}], "
               f"Δlatency {vs_int8['d_lat']:+.2f} ms [{vs_int8['d_lat_ci'][0]:+.2f}, {vs_int8['d_lat_ci'][1]:+.2f}]",
               (f"- coin flip INT8↔FP32 at the same mean latency: {mix:.1f}% → v2 is {chosen_test['acc']-mix:+.1f} pp"
                if mix is not None else
                "- v2 is faster than every point of the original INT8↔FP32 design space"),
               ""]

    # mixed schedule (original protocol shape: 40% idle then 60% stressed)
    if len(conds) == 2:
        n = len(test)
        tiers = ["idle"] * int(0.4 * n) + ["stressed"] * (n - int(0.4 * n))
        pol = ResourceAwareCascade(op_points)
        mixed, _, _ = simulate_policy(T, Ls, pol, test, tiers)
        statics_mixed = {}
        for v in (BASE, INT8):
            corr = T["correct"][v][test]
            lat = np.array([Ls[t][v][i] for t, i in zip(tiers, test)])
            statics_mixed[v] = _summ(corr, lat)
        summary["mixed_schedule"] = {"v2": mixed, "static": statics_mixed}
        md += ["## Mixed schedule (40% idle → 60% stressed, like the original protocol)", "",
               "| config | acc % | mean ms | p95 ms |", "|---|---|---|---|",
               f"| Baseline fp32_224 | {statics_mixed[BASE]['acc']:.1f} | {statics_mixed[BASE]['lat']:.2f} | {statics_mixed[BASE]['p95']:.2f} |",
               f"| AlwaysINT8 int8_224 | {statics_mixed[INT8]['acc']:.1f} | {statics_mixed[INT8]['lat']:.2f} | {statics_mixed[INT8]['p95']:.2f} |",
               f"| **v2 resource-aware** | **{mixed['acc']:.1f}** | **{mixed['lat']:.2f}** | {mixed['p95']:.2f} |", ""]

    # reference policies for the live comparison (chosen on the TUNE half, same budgets)
    statics = {c: [v for v in T["cands"]] for c in conds}
    best_static = {}
    for ci, cond in enumerate(conds):
        L = Ls[cond]
        feas = [v for v in statics[cond] if static_eval(T, L, v, tune)["lat"] <= op_points[cond]["budget_ms"]]
        best_static[cond] = (max(feas, key=lambda v: static_eval(T, L, v, tune)["acc"]) if feas
                             else min(statics[cond], key=lambda v: static_eval(T, L, v, tune)["lat"]))
    worst = conds[-1]   # 'stressed' when both exist: one fixed model must fit the budget there too
    fixed = [v for v in T["cands"] if all(static_eval(T, Ls[c], v, tune)["lat"] <= op_points[c]["budget_ms"] for c in conds)]
    fixed_model = (max(fixed, key=lambda v: static_eval(T, Ls[worst], v, tune)["acc"]) if fixed
                   else min(T["cands"], key=lambda v: static_eval(T, Ls[worst], v, tune)["lat"]))
    op_points["_reference"] = {"best_static_per_state": best_static, "fixed_model": fixed_model}

    op_points["_meta"] = {"source_table": a.table, "profiles": {c: getattr(a, f"profile_{c}") for c in conds},
                          "generated": summary["generated"], "label": a.label,
                          "objective": "max accuracy s.t. mean latency <= budget (tuned on TUNE split)"}
    summary["operating_points"] = op_points
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    if a.write_config:
        path = assert_inside_v2(a.config_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(op_points, f, indent=2)
        print(f"\noperating points -> {path}")
    print(f"report -> {os.path.join(out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
