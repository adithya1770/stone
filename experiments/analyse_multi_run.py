import csv
import os
import sys
import statistics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.reward import calculate_reward


RUN_DIRS = [
    "results/final_eightsignal_run1",
    "results/final_eightsignal_run2",
    "results/final_eightsignal_run3",
]

FILES = {
    "Baseline":         "baseline_log.csv",
    "AlwaysINT8":       "always_int8_log.csv",
    "LinUCB Cold":      "linucb_cold_log.csv",
    "LinUCB Warm":      "linucb_warm_log.csv",
    "EightSignal Cold": "eightsignal_cold_log.csv",
    "EightSignal Warm": "eightsignal_warm_log.csv",
}

STRESS_THRESHOLD = 70.0
MAX_PLAUSIBLE_LATENCY_MS = 500.0


def read_csv(path):
    rows = []
    skipped = 0
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            latency = float(row["latency_ms"])
            if latency > MAX_PLAUSIBLE_LATENCY_MS:
                skipped += 1
                continue
            rows.append({
                "cpu":              float(row["cpu"]),
                "ram":              float(row["ram"]),
                "temperature":      float(row["temperature"]),
                "latency_ms":       latency,
                "confidence":       float(row["confidence"]),
                "model":            row["model"],
                "decision_time_ms": float(row.get("decision_time_ms", 0) or 0),
                "correct":          row.get("correct", "")
            })
    if skipped:
        print(f"Warning: {path} — skipped {skipped} row(s) with "
              f"latency_ms > {MAX_PLAUSIBLE_LATENCY_MS}ms "
              f"(likely system sleep/interruption, not real inference time)")
    return rows


def add_reward(rows):
    for r in rows:
        r["reward"] = calculate_reward(
            confidence=r["confidence"],
            latency_ms=r["latency_ms"],
            cpu=r["cpu"],
            ram=r["ram"],
            temperature=r["temperature"],
            decision_time_ms=r["decision_time_ms"]
        )
    return rows


def get_run_lengths(rows):
    if not rows:
        return []
    run_lengths = []
    current_len = 1
    for i in range(1, len(rows)):
        if rows[i]["model"] == rows[i - 1]["model"]:
            current_len += 1
        else:
            run_lengths.append(current_len)
            current_len = 1
    run_lengths.append(current_len)
    return run_lengths


def analyse(rows):
    rows = add_reward(rows)

    normal   = [r for r in rows if r["cpu"] < STRESS_THRESHOLD]
    stressed = [r for r in rows if r["cpu"] >= STRESS_THRESHOLD]

    def avg(values):
        return sum(values) / len(values) if values else 0

    def avg_latency(subset):
        return avg([r["latency_ms"] for r in subset])

    def avg_confidence(subset):
        return avg([r["confidence"] for r in subset])

    def avg_reward(subset):
        return avg([r["reward"] for r in subset])

    def max_latency(subset):
        return max([r["latency_ms"] for r in subset]) if subset else 0

    def accuracy_pct(subset):
        known = [r for r in subset if r["correct"] in ("True", "False")]
        if not known:
            return None
        hits = sum(1 for r in known if r["correct"] == "True")
        return 100 * hits / len(known)

    switches = sum(
        1 for i in range(1, len(rows))
        if rows[i]["model"] != rows[i-1]["model"]
    )

    run_lengths = get_run_lengths(rows)
    blip_runs = [l for l in run_lengths if l == 1]
    sustained_runs = [l for l in run_lengths if l >= 2]

    exploration_blips = len(blip_runs)
    sustained_switches = max(len(sustained_runs) - 1, 0)

    int8_count = sum(1 for r in stressed if r["model"] == "int8")
    int8_pct = 100 * int8_count / len(stressed) if stressed else 0

    return {
        "avg_latency_normal":      avg_latency(normal),
        "avg_latency_stressed":    avg_latency(stressed),
        "max_latency_stressed":    max_latency(stressed),
        "avg_confidence_normal":   avg_confidence(normal),
        "avg_confidence_stressed": avg_confidence(stressed),
        "avg_reward_normal":       avg_reward(normal),
        "avg_reward_stressed":     avg_reward(stressed),
        "model_switches":          switches,
        "sustained_switches":      sustained_switches,
        "exploration_blips":       exploration_blips,
        "int8_under_stress_pct":   int8_pct,
        "avg_decision_time_ms":    avg([r["decision_time_ms"] for r in rows]),
        "accuracy_pct":            accuracy_pct(rows),
        "accuracy_pct_normal":     accuracy_pct(normal),
        "accuracy_pct_stressed":   accuracy_pct(stressed),
    }


per_run_results = {name: [] for name in FILES}

for run_dir in RUN_DIRS:
    for name, fname in FILES.items():
        path = os.path.join(run_dir, fname)
        if os.path.exists(path):
            per_run_results[name].append(analyse(read_csv(path)))
        else:
            print(f"Missing: {path}")


def fmt_stat(values, unit=""):
    known = [v for v in values if v is not None]
    if not known:
        return "n/a"
    mean = statistics.mean(known)
    std = statistics.stdev(known) if len(known) > 1 else 0
    return f"{mean:6.2f}{unit}  ±  {std:5.2f}{unit}"


metrics = [
    ("Avg latency — normal",         "avg_latency_normal",      "ms"),
    ("Avg latency — stressed",       "avg_latency_stressed",    "ms"),
    ("Max latency — stressed",       "max_latency_stressed",    "ms"),
    ("Avg confidence — normal",      "avg_confidence_normal",   ""),
    ("Avg confidence — stressed",    "avg_confidence_stressed", ""),
    ("Avg reward — normal",          "avg_reward_normal",       ""),
    ("Avg reward — stressed",        "avg_reward_stressed",     ""),
    ("Accuracy — overall",           "accuracy_pct",            "%"),
    ("Accuracy — normal",            "accuracy_pct_normal",     "%"),
    ("Accuracy — stressed",          "accuracy_pct_stressed",   "%"),
    ("Model switches (raw)",         "model_switches",          ""),
    ("  of which exploration blips", "exploration_blips",       ""),
    ("Sustained switches",           "sustained_switches",      ""),
    ("INT8 usage under stress",      "int8_under_stress_pct",   "%"),
    ("Avg decision time",            "avg_decision_time_ms",    "ms"),
]

print("\n" + "=" * 100)
print(f"STONE — FINAL 4-EXPERIMENT RESULTS ACROSS {len(RUN_DIRS)} RUNS (mean ± std)")
print("=" * 100)

for name in FILES:
    runs = per_run_results[name]
    if not runs:
        continue
    print(f"\n{name}  ({len(runs)} runs)")
    print("-" * 60)
    for label, key, unit in metrics:
        values = [r[key] for r in runs]
        print(f"  {label:<28}: {fmt_stat(values, unit)}   "
              f"(runs: {[round(v,2) if v is not None else None for v in values]})")

print("\n" + "=" * 100)