import csv
import os


FILES = {
    "Baseline":     "logging/baseline_log.csv",
    "Rule-Based":   "logging/rule_based_log.csv",
    "LinUCB Cold":  "logging/linucb_log.csv",
    "LinUCB Warm":  "logging/linucb_warm_log.csv"
}

STRESS_THRESHOLD = 70.0


def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "cpu":        float(row["cpu"]),
                "latency_ms": float(row["latency_ms"]),
                "confidence": float(row["confidence"]),
                "model":      row["model"]
            })
    return rows


def analyse(rows):
    normal   = [r for r in rows if r["cpu"] < STRESS_THRESHOLD]
    stressed = [r for r in rows if r["cpu"] >= STRESS_THRESHOLD]

    def avg(values):
        return round(sum(values) / len(values), 2) if values else 0

    def avg_latency(subset):
        return avg([r["latency_ms"] for r in subset])

    def avg_confidence(subset):
        return avg([r["confidence"] for r in subset])

    def max_latency(subset):
        return round(max([r["latency_ms"] for r in subset]), 2) if subset else 0

    switches = sum(
        1 for i in range(1, len(rows))
        if rows[i]["model"] != rows[i-1]["model"]
    )

    int8_count = sum(1 for r in stressed if r["model"] == "int8")
    int8_pct = round(100 * int8_count / len(stressed), 1) if stressed else 0

    return {
        "total_readings":            len(rows),
        "normal_readings":           len(normal),
        "stressed_readings":         len(stressed),
        "avg_latency_normal":        avg_latency(normal),
        "avg_latency_stressed":      avg_latency(stressed),
        "max_latency_stressed":      max_latency(stressed),
        "avg_confidence_normal":     avg_confidence(normal),
        "avg_confidence_stressed":   avg_confidence(stressed),
        "model_switches":            switches,
        "int8_under_stress_pct":     int8_pct
    }


results = {}
for name, path in FILES.items():
    if os.path.exists(path):
        results[name] = analyse(read_csv(path))
    else:
        print(f"Missing: {path}")


col_w   = 15
label_w = 34
names   = list(results.keys())

print("\n" + "=" * (label_w + col_w * len(names)))
print("STONE — EXPERIMENT RESULTS")
print("=" * (label_w + col_w * len(names)))

header = f"{'Metric':<{label_w}}" + "".join(f"{n:>{col_w}}" for n in names)
print(header)
print("-" * (label_w + col_w * len(names)))

metrics = [
    ("Total readings",               "total_readings",          ""),
    ("Normal readings (cpu<70%)",    "normal_readings",         ""),
    ("Stressed readings (cpu≥70%)",  "stressed_readings",       ""),
    ("Avg latency — normal",         "avg_latency_normal",      "ms"),
    ("Avg latency — stressed",       "avg_latency_stressed",    "ms"),
    ("Max latency — stressed",       "max_latency_stressed",    "ms"),
    ("Avg confidence — normal",      "avg_confidence_normal",   ""),
    ("Avg confidence — stressed",    "avg_confidence_stressed", ""),
    ("Model switches",               "model_switches",          ""),
    ("INT8 usage under stress",      "int8_under_stress_pct",   "%"),
]

for label, key, unit in metrics:
    row = f"{label:<{label_w}}"
    for name in names:
        val  = results[name][key]
        cell = f"{val}{unit}"
        row += f"{cell:>{col_w}}"
    print(row)

print("=" * (label_w + col_w * len(names)))

print("\nKEY FINDINGS:")

if "Baseline" in results:
    base_lat = results["Baseline"]["avg_latency_stressed"]

    for name in ["Rule-Based", "LinUCB Cold", "LinUCB Warm"]:
        if name in results:
            lat = results[name]["avg_latency_stressed"]
            pct = round(100 * (base_lat - lat) / base_lat, 1)
            print(f"  {name:<14} reduces stressed latency by {pct}% vs Baseline")

if "LinUCB Cold" in results and "LinUCB Warm" in results:
    cold_lat = results["LinUCB Cold"]["avg_latency_stressed"]
    warm_lat = results["LinUCB Warm"]["avg_latency_stressed"]
    pct = round(100 * (cold_lat - warm_lat) / cold_lat, 1)
    print(f"  LinUCB Warm    reduces stressed latency by {pct}% vs LinUCB Cold")

if "Rule-Based" in results and "LinUCB Warm" in results:
    rb_lat   = results["Rule-Based"]["avg_latency_stressed"]
    warm_lat = results["LinUCB Warm"]["avg_latency_stressed"]
    pct = round(100 * (rb_lat - warm_lat) / rb_lat, 1)
    print(f"  LinUCB Warm    reduces stressed latency by {pct}% vs Rule-Based")

if "LinUCB Cold" in results and "LinUCB Warm" in results:
    cold_sw = results["LinUCB Cold"]["model_switches"]
    warm_sw = results["LinUCB Warm"]["model_switches"]
    print(f"\n  LinUCB Cold switches: {cold_sw}  —  LinUCB Warm switches: {warm_sw}")
    print(f"  Warm start makes more decisive switching decisions from inference 1")

print()