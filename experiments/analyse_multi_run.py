import csv
import os
import statistics


RUN_DIRS = ["logging_run1", "logging_run2", "logging_run3"]

FILES = {
    "Baseline":         "baseline_log.csv",
    "AlwaysINT8":       "always_int8_log.csv",
    "Rule-Based":       "rule_based_log.csv",
    "EpsilonGreedy":    "egreedy_log.csv",
    "LinUCB Cold":      "linucb_log.csv",
    "LinUCB Warm":      "linucb_warm_log.csv",
    "Thompson":         "thompson_log.csv",
    "SlidingLinUCBWarm": "sliding_log_warm.csv",
    "SlidingLinUCBCold": "sliding_log_cold.csv"
}

STRESS_THRESHOLD = 70.0


def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "cpu":              float(row["cpu"]),
                "latency_ms":       float(row["latency_ms"]),
                "confidence":       float(row["confidence"]),
                "model":            row["model"],
                "decision_time_ms": float(row.get("decision_time_ms", 0) or 0)
            })
    return rows


def analyse(rows):
    normal   = [r for r in rows if r["cpu"] < STRESS_THRESHOLD]
    stressed = [r for r in rows if r["cpu"] >= STRESS_THRESHOLD]

    def avg(values):
        return sum(values) / len(values) if values else 0

    def avg_latency(subset):
        return avg([r["latency_ms"] for r in subset])

    def avg_confidence(subset):
        return avg([r["confidence"] for r in subset])

    def max_latency(subset):
        return max([r["latency_ms"] for r in subset]) if subset else 0

    switches = sum(
        1 for i in range(1, len(rows))
        if rows[i]["model"] != rows[i-1]["model"]
    )

    int8_count = sum(1 for r in stressed if r["model"] == "int8")
    int8_pct = 100 * int8_count / len(stressed) if stressed else 0

    return {
        "avg_latency_normal":      avg_latency(normal),
        "avg_latency_stressed":    avg_latency(stressed),
        "max_latency_stressed":    max_latency(stressed),
        "avg_confidence_normal":   avg_confidence(normal),
        "avg_confidence_stressed": avg_confidence(stressed),
        "model_switches":          switches,
        "int8_under_stress_pct":   int8_pct,
        "avg_decision_time_ms":    avg([r["decision_time_ms"] for r in rows])
    }


per_run_results = {name: [] for name in FILES}

for run_dir in RUN_DIRS:
    for name, fname in FILES.items():
        path = os.path.join(run_dir, fname)
        if os.path.exists(path):
            per_run_results[name].append(analyse(read_csv(path)))
        else:
            print(f"Missing: {path}")

metrics = [
    ("Avg latency — normal",      "avg_latency_normal",      "ms"),
    ("Avg latency — stressed",    "avg_latency_stressed",    "ms"),
    ("Max latency — stressed",    "max_latency_stressed",    "ms"),
    ("Avg confidence — normal",   "avg_confidence_normal",   ""),
    ("Avg confidence — stressed", "avg_confidence_stressed", ""),
    ("Model switches",            "model_switches",          ""),
    ("INT8 usage under stress",   "int8_under_stress_pct",   "%"),
    ("Avg decision time",         "avg_decision_time_ms",    "ms"),
]

print("\n" + "=" * 100)
print(f"STONE — AVERAGED RESULTS ACROSS {len(RUN_DIRS)} RUNS (mean ± std)")
print("=" * 100)

for name in FILES:
    runs = per_run_results[name]
    if not runs:
        continue
    print(f"\n{name}  ({len(runs)} runs)")
    print("-" * 60)
    for label, key, unit in metrics:
        values = [r[key] for r in runs]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        print(f"  {label:<28}: {mean:6.2f}{unit}  ± {std:5.2f}{unit}   "
              f"(runs: {[round(v,2) for v in values]})")

print("\n" + "=" * 100)