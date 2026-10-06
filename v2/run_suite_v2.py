"""Re-run the ORIGINAL Raspberry Pi protocol with the v2 router added.

    python -m v2.run_suite_v2 --trials 3

Protocol (same constants as experiments/run_master_suite_multitrial.py,
which produced Table I of the paper):
  - images: runtime.image_pool.ImagePool(seed=42), first 100, fixed order
  - telemetry: runtime.telemetry.get_telemetry() before every request
    (the same blocking 0.5 s reading the original controllers received)
  - stress: `stress-ng --cpu 0 --timeout 60s` started at iteration 40
  - "stressed" rows: cpu >= 70 (same filter as analyse_pi_multitrial.py)

Configs (each trial, in this order, with a cool-down between configs):
  baseline           static fp32_224                 (original)
  always_int8        static int8_224                 (original)
  linucb_cold        original LinUCB, fresh state    (original code, imported)
  eightsignal_cold   original EightSignal, fresh     (original code, imported)
  v2_cascade         input-aware only (fixed stressed ladder, fixed threshold)
  v2_resource_aware  full v2: tier -> ladder, margin -> escalate, online budget
  --with-fixed adds: baseline_fixed (fp32n_224) and v2_resource_aware_fixed
                     (uses configs/operating_points_fixed.json)
  --with-final adds: eightsignal_rebuilt  original EightSignal, unchanged, choosing between the
                                          rebuilt INT8 / FP32 files
                     v2_final             EightSignal decides the device state + per-photo check
                                          (uses configs/operating_points_final.json)

Original controllers' state files go to v2/results_v2/... - nothing outside
v2/ is written. Logs keep the original CSV columns plus v2 extras.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from datetime import datetime

import psutil

from v2.common.paths import CONFIG_ROOT, RESULTS_ROOT, assert_inside_v2, enter_repo
from v2.common.engine import MultiVariantEngine, load_rgb
from v2.routing.bandits import EpsilonGreedyLadder
from v2.routing.eightsignal_cascade import EightSignalCascade
from v2.routing.policies import FixedCascade, ResourceAwareCascade, Static, his_controller_policy
from v2.routing.resource_state import ResourceMonitor
from v2.routing.telemetry import read_throttled
from v2.bench.stress import Stress

STRESS_START_AT = 40
STRESS_DURATION = 60
ITERATIONS = 100
BUDGET_MODE = "relative"
LADDER_TARGET = None    # --ladder-target-ms: fixed time budget (ms) for the absolute ladder configs
REBUILT_MAP = {"fp32": "mnv2f_224", "int8": "mnv2q_224"}   # his two model slots -> the rebuilt files
FRESH_STATE = ("egreedy", "eightsignal_rebuilt", "v2_final", "eightsignal_ladder", "v2_ladder", "v2_ladder_8sig",
               "linucb_ladder")
LADDER_ARMS = ["mnv2q_224", "efl0q_224", "efl1q_240", "efl2q_260", "efl3q_280", "efl4q_300"]  # (+ every *_cold) start fresh each trial

HIS_FIELDS = ["timestamp", "cpu", "ram", "temperature", "battery", "health_score", "model",
              "decision_source", "image_path", "true_label", "label", "correct", "confidence",
              "latency_ms", "decision_time_ms"]
V2_FIELDS = ["iteration", "config", "trial", "tier", "first_variant", "final_variant", "escalated",
             "margin_first", "latency_first_ms", "latency_second_ms", "preprocess_ms", "e2e_ms",
             "cpu_time_ms", "throttled", "theta", "rss_mb"]


def load_op(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}\nRun bench/replay_eval.py --write-config first (see v2/pi_run_all.sh).")
    with open(path) as f:
        return json.load(f)


def _warm(kind, state_dir):
    """Original 'warm' semantics (run.sh): continue from the state the cold run
    of the SAME trial just learned. Cold and warm always run back to back."""
    cold = os.path.join(state_dir, f"{kind}_cold_state.npz")
    warm = os.path.join(state_dir, f"{kind}_warm_state.npz")
    if os.path.exists(cold):
        shutil.copy(cold, assert_inside_v2(warm))
    elif os.path.exists(warm):
        os.remove(assert_inside_v2(warm))  # no cold run this trial -> starts fresh
    return his_controller_policy(kind, warm)


def _mapped(pol, vmap):
    """Original controller, unchanged, but its 'fp32'/'int8' slots run other model files."""
    pol.variant_map = dict(vmap)
    return pol


def make_configs(op, op_fixed, state_dir, with_fixed, with_8sig=False, with_warm=False, op_rebuilt=None,
                 op_final=None, op_ladder=None):
    st = op["stressed"] if "stressed" in op else op["idle"]
    cfgs = [
        ("baseline", lambda: Static("fp32_224", "baseline")),
        ("always_int8", lambda: Static("int8_224", "always_int8")),
        ("linucb_cold", lambda: his_controller_policy("linucb", os.path.join(state_dir, "linucb_cold_state.npz"))),
    ] + ([
        ("linucb_warm", lambda: _warm("linucb", state_dir)),
        ("egreedy", lambda: his_controller_policy("egreedy", os.path.join(state_dir, "egreedy_state.npz"))),
    ] if with_warm else []) + [
        ("eightsignal_cold", lambda: his_controller_policy("eightsignal", os.path.join(state_dir, "eightsignal_cold_state.npz"))),
    ] + ([
        ("eightsignal_warm", lambda: _warm("eightsignal", state_dir)),
    ] if with_warm else []) + [
        ("v2_cascade", lambda: (FixedCascade(st["first"], st["second"], st["theta"], "v2_cascade", st.get("signal", "margin"))
                                if st.get("second") else Static(st["first"], "v2_cascade"))),
        ("v2_resource_aware", lambda: ResourceAwareCascade(op, name="v2_resource_aware", budget_mode=BUDGET_MODE)),
    ]
    if with_fixed:
        cfgs += [("baseline_fixed", lambda: Static("fp32n_224", "baseline_fixed")),
                 ("v2_resource_aware_fixed", lambda: ResourceAwareCascade(op_fixed, name="v2_resource_aware_fixed",
                                                                          budget_mode=BUDGET_MODE))]
    if op_rebuilt is not None:
        # models rebuilt from the Keras weights with correct preprocessing and
        # INT8 calibration (variants/build_variants.py); reported separately
        cfgs += [("always_int8_rebuilt", lambda: Static("mnv2q_224", "always_int8_rebuilt")),
                 ("baseline_rebuilt", lambda: Static("mnv2f_224", "baseline_rebuilt")),
                 ("v2_rebuilt", lambda: ResourceAwareCascade(op_rebuilt, name="v2_rebuilt", budget_mode=BUDGET_MODE))]
    if with_8sig and with_fixed:
        # the ORIGINAL EightSignal vote decides the device tier (resource side);
        # v2's per-image confidence check decides escalation (input side)
        def _v2_8sig():
            p = ResourceAwareCascade(op_fixed, name="v2_fixed_8signal", budget_mode=BUDGET_MODE)
            p.monitor_mode = "eightsignal"
            return p
        cfgs.append(("v2_fixed_8signal", _v2_8sig))
    if op_final is not None:
        # the final system and its fair comparison: both use the group's
        # EightSignal controller and the rebuilt models; v2_final adds the
        # per-photo check and per-state time budgets
        cfgs += [("eightsignal_rebuilt", lambda: _mapped(
                     his_controller_policy("eightsignal", os.path.join(state_dir, "eightsignal_rebuilt_state.npz")),
                     REBUILT_MAP)),
                 ("v2_final", lambda: EightSignalCascade(op_final, os.path.join(state_dir, "v2_final_state.npz"),
                                                         name="v2_final", budget_mode=BUDGET_MODE))]
        if op_rebuilt is None:
            cfgs += [("always_int8_rebuilt", lambda: Static("mnv2q_224", "always_int8_rebuilt")),
                     ("baseline_rebuilt", lambda: Static("mnv2f_224", "baseline_rebuilt"))]
    if op_ladder is not None:
        # model ladder (MobileNetV2 + EfficientNet-Lite0..4, all INT8) under a time budget per state
        ref = op_ladder["_reference"]
        best = ref["best_static_per_state"]
        cfgs += [("fixed_ladder", lambda: Static(ref["fixed_model"], "fixed_ladder")),          # one model that fits the budget even under stress
                 ("big_ladder", lambda: Static(best["idle"], "big_ladder")),                   # best model for idle, kept under stress
                 ("eightsignal_ladder", lambda: _mapped(                                     # EightSignal switches idle-model <-> stressed-model
                     his_controller_policy("eightsignal", os.path.join(state_dir, "eightsignal_ladder_state.npz")),
                     {"fp32": best["idle"], "int8": best["stressed"]})),
                 ("v2_ladder", lambda: EightSignalCascade(op_ladder, os.path.join(state_dir, "v2_ladder_state.npz"),
                                                          name="v2_ladder", budget_mode=BUDGET_MODE))]
        # Second ladder run: the time budget is a fixed SLA (absolute ms) in every state, and the
        # device state comes either from EightSignal or from v2's simple CPU/thermal monitor.
        op_abs = json.loads(json.dumps(op_ladder))
        if LADDER_TARGET:
            for t, o in op_abs.items():
                if not t.startswith("_"):
                    o["budget_ms"] = float(LADDER_TARGET)
        device_only = {t: {"first": best[t], "second": None, "budget_ms": op_abs[t]["budget_ms"],
                           "est_ms": op_abs[t].get("est_ms", {})} for t in best}
        cfgs += [("cpu_ladder", lambda: ResourceAwareCascade(device_only, name="cpu_ladder", budget_mode="absolute")),
                 ("v2_ladder_8sig", lambda: EightSignalCascade(op_abs, os.path.join(state_dir, "v2_ladder_8sig_state.npz"),
                                                               name="v2_ladder_8sig", budget_mode="absolute")),
                 ("v2_ladder_cpu", lambda: ResourceAwareCascade(op_abs, name="v2_ladder_cpu", budget_mode="absolute")),
                 # the best single model at about v2's average time (fairest single-model comparison)
                 ("lite1_ladder", lambda: Static("efl1q_240", "lite1_ladder")),
                 # the original LinUCB (unchanged) switching between the best idle / best stressed model
                 ("linucb_ladder", lambda: _mapped(
                     his_controller_policy("linucb", os.path.join(state_dir, "linucb_ladder_state.npz")),
                     {"fp32": best["idle"], "int8": best["stressed"]})),
                 # EdgeMLBalancer-style epsilon-greedy over all ladder models (routing/bandits.py)
                 ("egreedy_ladder", lambda: EpsilonGreedyLadder(
                     LADDER_ARMS, budget_ms=op_abs["stressed"]["budget_ms"], epsilon=0.1, lam=1.0, seed=0))]
    return cfgs


def run_config(name, factory, items, eng, out_csv, trial, cpu_hi, stressor,
               stress_start=STRESS_START_AT, stress_duration=STRESS_DURATION):
    from runtime.telemetry import get_telemetry          # original, read-only
    from runtime.reward import calculate_reward          # original, read-only

    if name.endswith("_cold") or name in FRESH_STATE:  # cold = fresh state every trial
        sp = os.path.join(os.path.dirname(out_csv), f"{name}_state.npz")
        if os.path.exists(sp):
            os.remove(assert_inside_v2(sp))
    pol = factory()
    is_his = hasattr(pol, "decide")
    has_device_layer = hasattr(pol, "device_tier")   # v2_final: EightSignal decides the tier
    vmap = getattr(pol, "variant_map", {})
    monitor = ResourceMonitor(mode=getattr(pol, "monitor_mode", "simple"), cpu_hi=cpu_hi)
    proc = psutil.Process()
    stress = None
    rows = []
    print(f"  === {name} ===", flush=True)
    for i, (path, _, true_idx) in enumerate(items):
        if i == stress_start:
            stress = Stress(seconds=stress_duration, prefer=stressor).start(settle=1.5)
        t = get_telemetry()
        thr = read_throttled()
        t["throttled"] = thr
        t["throttled_now"] = bool(thr & 0xF) if thr is not None else False

        c0 = proc.cpu_times()
        w0 = time.perf_counter()
        img = load_rgb(path)
        r2 = None
        theta = ""
        tier = ""
        if is_his:
            d0 = time.time()
            m, scores, source = pol.decide(t)
            dms = round((time.time() - d0) * 1000, 3)
            r1 = eng.run_image(img, vmap.get(m, f"{m}_224"))
            reward = calculate_reward(confidence=r1["confidence"], latency_ms=r1["latency_ms"],
                                      cpu=t["cpu"], ram=t["ram"], temperature=t["temperature"],
                                      decision_time_ms=dms)
            pol.update(m, t, reward, r1["confidence"])
            model_col, health = (r1["variant"] if vmap else m), round(scores.get("fp32", 0), 4)
        else:
            d0 = time.time()
            tier = pol.device_tier(t) if has_device_layer else monitor.update(t)
            ctx = {"tier": tier}
            v1 = pol.first(ctx)
            dms = round((time.time() - d0) * 1000, 3)
            r1 = eng.run_image(img, v1)
            d1 = time.time()
            v2 = pol.second(ctx, r1)
            dms += round((time.time() - d1) * 1000, 3)
            if v2:
                r2 = eng.run_image(img, v2)
            total = r1["latency_ms"] + (r2["latency_ms"] if r2 else 0.0)
            pol.observe(ctx, [r1, r2], total)
            if has_device_layer:   # original EightSignal update, same reward function as the original
                fin = r2 or r1
                pol.feedback(t, calculate_reward(confidence=fin["confidence"], latency_ms=total, cpu=t["cpu"],
                                                 ram=t["ram"], temperature=t["temperature"], decision_time_ms=dms),
                             fin["confidence"])
            th = getattr(pol, "theta", None)
            th = th(ctx) if callable(th) else th
            theta = round(th, 4) if isinstance(th, float) else ""
            source = pol.name
            final = r2 or r1
            model_col, health = final["variant"], 0
        e2e = (time.perf_counter() - w0) * 1000
        c1 = proc.cpu_times()
        final = r2 or r1
        ok = final["index"] == true_idx
        rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu": t["cpu"], "ram": t["ram"], "temperature": t["temperature"], "battery": t["battery"],
            "health_score": health, "model": model_col, "decision_source": source,
            "image_path": path, "true_label": eng.labels[true_idx + 1] if true_idx is not None else "unknown",
            "label": final["label"], "correct": str(ok), "confidence": final["confidence"],
            "latency_ms": round(r1["latency_ms"] + (r2["latency_ms"] if r2 else 0.0), 3),
            "decision_time_ms": dms,
            "iteration": i, "config": name, "trial": trial, "tier": tier,
            "first_variant": r1["variant"], "final_variant": final["variant"], "escalated": str(r2 is not None),
            "margin_first": r1["margin"], "latency_first_ms": r1["latency_ms"],
            "latency_second_ms": r2["latency_ms"] if r2 else "",
            "preprocess_ms": round(r1["preprocess_ms"] + (r2["preprocess_ms"] if r2 else 0.0), 3),
            "e2e_ms": round(e2e, 3),
            "cpu_time_ms": round(((c1.user + c1.system) - (c0.user + c0.system)) * 1000, 3),
            "throttled": thr if thr is not None else "", "theta": theta,
            "rss_mb": round(proc.memory_info().rss / 1e6, 1),
        })
        if i % 20 == 0 or i == len(items) - 1:
            print(f"    [{i}] {model_col:<10} cpu={t['cpu']}% tier={tier or '-'} "
                  f"lat={rows[-1]['latency_ms']}ms ok={ok}", flush=True)
    if stress:
        stress.stop()
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HIS_FIELDS + V2_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"    saved -> {out_csv}")


def cool_down(min_s, max_temp, max_s):
    """Sleep at least min_s, then until temperature <= max_temp (or max_s)."""
    from v2.routing.telemetry import read_temperature
    t0 = time.time()
    time.sleep(min_s)
    temp = read_temperature()
    while temp is not None and temp > max_temp and time.time() - t0 < max_s:
        time.sleep(5)
        temp = read_temperature()
    if temp is not None:
        print(f"    (cool-down {time.time() - t0:.0f}s, start temp {temp:.1f} C)", flush=True)
    return temp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M"))
    ap.add_argument("--op-points", default=os.path.join(CONFIG_ROOT, "operating_points.json"))
    ap.add_argument("--op-points-fixed", default=os.path.join(CONFIG_ROOT, "operating_points_fixed.json"))
    ap.add_argument("--with-fixed", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="subset of config names")
    ap.add_argument("--cooldown", type=float, default=20.0, help="minimum seconds between configs")
    ap.add_argument("--cooldown-temp", type=float, default=65.0,
                    help="also wait until CPU temperature <= this (max --cooldown-max s); the Pi reaches "
                         "75-85 C in long runs, which would penalise whichever config runs last")
    ap.add_argument("--cooldown-max", type=float, default=240.0)
    ap.add_argument("--order", default="rotate", choices=["rotate", "fixed"],
                    help="rotate: each trial starts at a different config so none always runs hottest")
    ap.add_argument("--cpu-hi", type=float, default=70.0, help="tier threshold on the original cpu reading")
    ap.add_argument("--stressor", default="auto", choices=["auto", "stress-ng", "python"])
    ap.add_argument("--dataset", default="his_pool",
                    help="his_pool (original 100-image sequence), imagenette_fresh (imagenette val images never "
                         "used by the original protocol or by v2 tuning; see --exclude-table), or tiny (pipeline test)")
    ap.add_argument("--iterations", type=int, default=ITERATIONS,
                    help="images per config (original protocol: 100)")
    ap.add_argument("--stress-start", type=int, default=None,
                    help="image index where stress-ng starts (default: 40 for 100 images, else 40%% of --iterations)")
    ap.add_argument("--stress-duration", type=float, default=None,
                    help="seconds of stress (default: 60 for the original 100-image protocol, else until the end)")
    ap.add_argument("--exclude-table", nargs="*", default=[],
                    help="response-table CSV(s) whose images must not be used (keeps tuning images out of the test)")
    ap.add_argument("--with-warm", action="store_true",
                    help="also run the original linucb_warm, egreedy and eightsignal_warm (full Table I)")
    ap.add_argument("--with-rebuilt", action="store_true",
                    help="add always_int8_rebuilt, baseline_rebuilt, v2_rebuilt (needs variants/registry.json "
                         "and configs/operating_points_rebuilt.json)")
    ap.add_argument("--op-points-rebuilt", default=os.path.join(CONFIG_ROOT, "operating_points_rebuilt.json"))
    ap.add_argument("--with-final", action="store_true",
                    help="add v2_final (EightSignal + photo check, rebuilt models) and eightsignal_rebuilt "
                         "(needs configs/operating_points_final.json)")
    ap.add_argument("--op-points-final", default=os.path.join(CONFIG_ROOT, "operating_points_final.json"))
    ap.add_argument("--with-ladder", action="store_true",
                    help="add fixed_ladder, big_ladder, eightsignal_ladder, v2_ladder (configs/operating_points_ladder.json)")
    ap.add_argument("--op-points-ladder", default=os.path.join(CONFIG_ROOT, "operating_points_ladder.json"))
    ap.add_argument("--ladder-target-ms", type=float, default=None,
                    help="override the time budget of cpu_ladder / v2_ladder_8sig / v2_ladder_cpu (e.g. 145)")
    ap.add_argument("--with-8sig", action="store_true",
                    help="add v2_fixed_8signal: the original EightSignal vote as v2's resource monitor")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--budget-mode", default="relative", choices=["relative", "absolute"])
    a = ap.parse_args()
    enter_repo()
    global BUDGET_MODE, LADDER_TARGET
    BUDGET_MODE = a.budget_mode
    LADDER_TARGET = a.ladder_target_ms

    if a.stressor in ("auto", "stress-ng") and not shutil.which("stress-ng"):
        print("WARNING: stress-ng not found; using Python CPU burners instead (install: sudo apt install stress-ng)")

    from v2.common.datasets import get_source
    exclude = set()
    for t in a.exclude_table:
        with open(t) as f:
            exclude |= {r["image_id"] for r in csv.DictReader(f)}
    items = get_source(a.dataset, limit=a.iterations, seed=42, exclude=exclude)[:a.iterations]
    if len(items) < a.iterations:
        print(f"WARNING: only {len(items)} images available")
    original = a.iterations == ITERATIONS and a.stress_start is None and a.stress_duration is None
    stress_start = a.stress_start if a.stress_start is not None else (STRESS_START_AT if original else int(0.4 * len(items)))
    stress_duration = a.stress_duration if a.stress_duration is not None else (STRESS_DURATION if original else 24 * 3600)
    op = load_op(a.op_points)
    op_fixed = load_op(a.op_points_fixed) if a.with_fixed else None
    op_rebuilt = load_op(a.op_points_rebuilt) if a.with_rebuilt else None
    op_final = load_op(a.op_points_final) if a.with_final else None
    op_ladder = load_op(a.op_points_ladder) if a.with_ladder else None
    variants = {"fp32_224", "int8_224"}
    if op_rebuilt or op_final:
        variants |= {"mnv2q_224", "mnv2f_224"}
    if op_ladder:
        r = op_ladder["_reference"]
        variants |= {r["fixed_model"], *r["best_static_per_state"].values(), *LADDER_ARMS}
    for o in [op] + [x for x in (op_fixed, op_rebuilt, op_final, op_ladder) if x]:
        for k, v in o.items():
            if not k.startswith("_"):
                variants |= {x for x in (v.get("first"), v.get("second")) if x}
                variants |= set(v.get("est_ms", {}))
    if a.with_fixed:
        variants.add("fp32n_224")
    eng = MultiVariantEngine(sorted(variants), num_threads=a.threads)

    root = assert_inside_v2(os.path.join(RESULTS_ROOT, "suite", a.tag))
    meta = {"tag": a.tag, "trials": a.trials, "dataset": a.dataset, "op_points": op,
            "op_points_fixed": op_fixed, "op_points_rebuilt": op_rebuilt, "op_points_final": op_final, "op_points_ladder": op_ladder, "ladder_target_ms": a.ladder_target_ms, "cooldown": a.cooldown, "cpu_hi": a.cpu_hi,
            "variants_loaded": sorted(variants), "budget_mode": a.budget_mode,
            "iterations": len(items), "stress_start": stress_start,
            "stress_duration": stress_duration if stress_duration < 24 * 3600 else "until end",
            "excluded_images": len(exclude),
            "started": datetime.now().isoformat()}
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for trial in range(1, a.trials + 1):
        print(f"\n{'=' * 60}\nTRIAL {trial}/{a.trials}\n{'=' * 60}")
        out_dir = os.path.join(root, f"trial{trial}", "imagenette" if a.dataset == "his_pool" else os.path.basename(a.dataset))
        os.makedirs(out_dir, exist_ok=True)
        cfgs = [c for c in make_configs(op, op_fixed, out_dir, a.with_fixed, a.with_8sig, a.with_warm, op_rebuilt,
                                        op_final, op_ladder)
                if not a.only or c[0] in a.only]
        # rotate over units so a warm config always directly follows its cold run
        units = []
        for c in cfgs:
            if c[0].endswith("_warm") and units and units[-1][-1][0] == c[0].replace("_warm", "_cold"):
                units[-1].append(c)
            else:
                units.append([c])
        if a.order == "rotate" and units:
            k = (trial - 1) * max(1, len(units) // a.trials) % len(units)
            units = units[k:] + units[:k]
        cfgs = [c for u in units for c in u]
        for name, factory in cfgs:
            t_start = cool_down(a.cooldown, a.cooldown_temp, a.cooldown_max)
            meta.setdefault("start_temperature", {})[f"trial{trial}/{name}"] = t_start
            run_config(name, factory, items, eng, os.path.join(out_dir, f"{name}_log.csv"),
                       trial, a.cpu_hi, a.stressor, stress_start, stress_duration)
            with open(os.path.join(root, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
    print(f"\nAll trials complete -> {root}\nNext: python -m v2.analyse_v2 --suite {root}")


if __name__ == "__main__":
    main()
