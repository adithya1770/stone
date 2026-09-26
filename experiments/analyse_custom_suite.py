import csv
import os
import statistics

RESULTS_DIR = "results/custom_suite"
CONFIGS = ["baseline", "always_int8", "linucb_cold", "linucb_warm",
           "egreedy", "eightsignal_cold", "eightsignal_warm"]
STRESS_THRESHOLD = 70.0


def read_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def avg(vals):
    return round(statistics.mean(vals), 2) if vals else 0.0


def accuracy(rows):
    known = [r for r in rows if r["correct"] in ("True", "False")]
    if not known:
        return None
    return round(100 * sum(1 for r in known if r["correct"] == "True") / len(known), 2)


def analyse(rows):
    stressed = [r for r in rows if float(r["cpu"]) >= STRESS_THRESHOLD]
    normal = [r for r in rows if float(r["cpu"]) < STRESS_THRESHOLD]

    switches = sum(1 for i in range(1, len(rows)) if rows[i]["model"] != rows[i-1]["model"])
    int8_stressed = sum(1 for r in stressed if r["model"] == "int8")
    int8_pct = round(100 * int8_stressed / len(stressed), 1) if stressed else 0.0

    return {
        "n": len(rows),
        "normal_n": len(normal),
        "stressed_n": len(stressed),
        "latency_normal": avg([float(r["latency_ms"]) for r in normal]),
        "latency_stressed": avg([float(r["latency_ms"]) for r in stressed]),
        "accuracy_overall": accuracy(rows),
        "accuracy_normal": accuracy(normal),
        "accuracy_stressed": accuracy(stressed),
        "switches": switches,
        "int8_under_stress_pct": int8_pct,
        "avg_decision_time_ms": avg([float(r["decision_time_ms"]) for r in rows]),
    }


results = {}
for cfg in CONFIGS:
    path = os.path.join(RESULTS_DIR, f"{cfg}_log.csv")
    if os.path.exists(path):
        results[cfg] = analyse(read_csv(path))
    else:
        print(f"Missing: {path}")

print("\n" + "=" * 140)
print("CUSTOM-IMAGE SUITE — RESULTS")
print("=" * 140)

header = f"{'Metric':<30}" + "".join(f"{cfg:>16}" for cfg in results.keys())
print(header)
print("-" * 140)

metric_labels = [
    ("Total readings", "n"),
    ("Normal readings", "normal_n"),
    ("Stressed readings", "stressed_n"),
    ("Avg latency normal (ms)", "latency_normal"),
    ("Avg latency stressed (ms)", "latency_stressed"),
    ("Accuracy overall (%)", "accuracy_overall"),
    ("Accuracy normal (%)", "accuracy_normal"),
    ("Accuracy stressed (%)", "accuracy_stressed"),
    ("Model switches", "switches"),
    ("INT8 usage under stress (%)", "int8_under_stress_pct"),
    ("Avg decision time (ms)", "avg_decision_time_ms"),
]

for label, key in metric_labels:
    row = f"{label:<30}"
    for cfg in results:
        val = results[cfg][key]
        val_str = "n/a" if val is None else str(val)
        row += f"{val_str:>16}"
    print(row)

print("=" * 140 + "\n")