import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.reward import calculate_reward

STRESS_THRESHOLD = 70.0
MAX_PLAUSIBLE_LATENCY_MS = 500.0

SOURCES = {
    "Baseline":              ("results/historical_13algo_exploration", "baseline_log.csv"),
    "AlwaysINT8":            ("results/historical_13algo_exploration", "always_int8_log.csv"),
    "Rule-Based":            ("results/historical_13algo_exploration", "rule_based_log.csv"),
    "EpsilonGreedy":         ("results/historical_13algo_exploration", "egreedy_log.csv"),
    "LinUCB Cold":           ("results/historical_13algo_exploration", "linucb_log.csv"),
    "LinUCB Warm":           ("results/historical_13algo_exploration", "linucb_warm_log.csv"),
    "Thompson":              ("results/historical_13algo_exploration", "thompson_log.csv"),
    "SlidingLinUCB Warm":    ("results/historical_13algo_exploration", "sliding_log_warm.csv"),
    "SlidingLinUCB Cold":    ("results/historical_13algo_exploration", "sliding_log_cold.csv"),
    "RuleGatedLinUCB":       ("results/historical_13algo_exploration", "rule_gated_linucb_log.csv"),
    "BooleanBandit":         ("results/historical_13algo_exploration", "boolean_bandit_log.csv"),
    "DisagreementGate":      ("results/historical_13algo_exploration", "disagreement_gate_log.csv"),
    "AndGateSafetyNet":      ("results/historical_13algo_exploration", "andgate_log.csv"),
    "MetaController":        ("results/historical_13algo_exploration", "meta_controller_log.csv"),
    "EightSignal (FINAL)":   ("results/final_eightsignal_run1", "eightsignal_warm_log.csv"),
}


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
              f"latency_ms > {MAX_PLAUSIBLE_LATENCY_MS}ms")
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
        return round(sum(values) / len(values), 2) if values else 0

    def accuracy_pct(subset):
        known = [r for r in subset if r["correct"] in ("True", "False")]
        if not known:
            return None
        hits = sum(1 for r in known if r["correct"] == "True")
        return round(100 * hits / len(known), 2)

    switches = sum(1 for i in range(1, len(rows)) if rows[i]["model"] != rows[i-1]["model"])
    run_lengths = get_run_lengths(rows)
    sustained = [l for l in run_lengths if l >= 2]
    sustained_switches = max(len(sustained) - 1, 0)

    int8_count = sum(1 for r in stressed if r["model"] == "int8")
    int8_pct = round(100 * int8_count / len(stressed), 1) if stressed else 0

    return {
        "avg_latency_stressed":  avg([r["latency_ms"] for r in stressed]),
        "accuracy_pct":          accuracy_pct(rows),
        "accuracy_pct_stressed": accuracy_pct(stressed),
        "model_switches":        switches,
        "sustained_switches":    sustained_switches,
        "int8_under_stress_pct": int8_pct,
        "avg_decision_time_ms":  round(avg([r["decision_time_ms"] for r in rows]), 4),
    }


results = {}
for name, (folder, fname) in SOURCES.items():
    path = os.path.join(folder, fname)
    if os.path.exists(path):
        results[name] = analyse(read_csv(path))
    else:
        print(f"Missing: {path}")


col_w, label_w = 14, 24
names = list(results.keys())

print("\n" + "=" * (label_w + col_w * len(names)))
print("STONE — ALGORITHM PROGRESSION (13 explored -> final EightSignalController)")
print("=" * (label_w + col_w * len(names)))

header = f"{'Metric':<{label_w}}" + "".join(f"{n[:col_w-1]:>{col_w}}" for n in names)
print(header)
print("-" * (label_w + col_w * len(names)))

metrics = [
    ("Latency (stressed)",      "avg_latency_stressed",  "ms"),
    ("Accuracy overall",        "accuracy_pct",          "%"),
    ("Accuracy stressed",       "accuracy_pct_stressed", "%"),
    ("Switches (raw)",          "model_switches",        ""),
    ("Sustained switches",      "sustained_switches",    ""),
    ("INT8 under stress",       "int8_under_stress_pct", "%"),
    ("Decision time",           "avg_decision_time_ms",  "ms"),
]

for label, key, unit in metrics:
    row = f"{label:<{label_w}}"
    for name in names:
        val = results[name][key]
        cell = f"{val}{unit}" if val is not None else "n/a"
        row += f"{cell:>{col_w}}"
    print(row)

print("=" * (label_w + col_w * len(names)))
print()