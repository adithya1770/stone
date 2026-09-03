import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.reward import calculate_reward


FILES = {
    "Baseline":                  "logging/baseline_log.csv",
    "AlwaysINT8":                "logging/always_int8_log.csv",
    "LinUCB Cold":                "logging/linucb_cold_log.csv",
    "LinUCB Warm":                "logging/linucb_warm_log.csv",
    "EightSignal Cold":           "logging/eightsignal_cold_log.csv",
    "EightSignal Warm":           "logging/eightsignal_warm_log.csv",
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
    """Group consecutive rows with the same model into 'runs' and return
    their lengths in order. E.g. int8,int8,fp32,int8,int8,int8 -> [2,1,3]"""
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

    def avg_latency(subset):
        return avg([r["latency_ms"] for r in subset])

    def avg_confidence(subset):
        return avg([r["confidence"] for r in subset])

    def avg_reward(subset):
        return round(avg([r["reward"] for r in subset]), 4)

    def max_latency(subset):
        return round(max([r["latency_ms"] for r in subset]), 2) if subset else 0

    def accuracy_pct(subset):
        known = [r for r in subset if r["correct"] in ("True", "False")]
        if not known:
            return None
        hits = sum(1 for r in known if r["correct"] == "True")
        return round(100 * hits / len(known), 2)

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
        "avg_reward_normal":         avg_reward(normal),
        "avg_reward_stressed":       avg_reward(stressed),
        "model_switches":            switches,
        "sustained_switches":        sustained_switches,
        "exploration_blips":         exploration_blips,
        "int8_under_stress_pct":     int8_pct,
        "avg_decision_time_ms":      round(avg([r["decision_time_ms"] for r in rows]), 4),
        "accuracy_pct":              accuracy_pct(rows),
        "accuracy_pct_normal":       accuracy_pct(normal),
        "accuracy_pct_stressed":     accuracy_pct(stressed),
    }


results = {}
for name, path in FILES.items():
    if os.path.exists(path):
        results[name] = analyse(read_csv(path))
    else:
        print(f"Missing: {path}")


col_w   = 20
label_w = 34
names   = list(results.keys())

print("\n" + "=" * (label_w + col_w * len(names)))
print("STONE — FINAL 4-EXPERIMENT RESULTS")
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

for label, key, unit in metrics:
    row = f"{label:<{label_w}}"
    for name in names:
        val  = results[name][key]
        cell = f"{val}{unit}" if val is not None else "n/a"
        row += f"{cell:>{col_w}}"
    print(row)

print("=" * (label_w + col_w * len(names)))

print("\nKEY FINDINGS:")

if "Baseline" in results:
    base_lat = results["Baseline"]["avg_latency_stressed"]
    for name in ["LinUCB Cold", "LinUCB Warm", "EightSignalController"]:
        if name in results:
            lat = results[name]["avg_latency_stressed"]
            pct = round(100 * (base_lat - lat) / base_lat, 1)
            print(f"  {name:<24} reduces stressed latency by {pct}% vs Baseline")

if "AlwaysINT8" in results and "LinUCB Warm" in results:
    int8_lat = results["AlwaysINT8"]["avg_latency_stressed"]
    warm_lat = results["LinUCB Warm"]["avg_latency_stressed"]
    int8_acc = results["AlwaysINT8"]["accuracy_pct"]
    warm_acc = results["LinUCB Warm"]["accuracy_pct"]
    print(f"\n  AlwaysINT8:  {int8_lat}ms stressed latency, {int8_acc}% overall accuracy")
    print(f"  LinUCB Warm: {warm_lat}ms stressed latency, {warm_acc}% overall accuracy")

if "LinUCB Cold" in results and "LinUCB Warm" in results:
    lc_lat  = results["LinUCB Cold"]["avg_latency_stressed"]
    lc_int8 = results["LinUCB Cold"]["int8_under_stress_pct"]
    lc_sw   = results["LinUCB Cold"]["model_switches"]
    lc_acc  = results["LinUCB Cold"]["accuracy_pct"]
    lw_lat  = results["LinUCB Warm"]["avg_latency_stressed"]
    lw_int8 = results["LinUCB Warm"]["int8_under_stress_pct"]
    lw_sw   = results["LinUCB Warm"]["model_switches"]
    lw_acc  = results["LinUCB Warm"]["accuracy_pct"]
    print(f"\n  LinUCB Cold: {lc_lat}ms stressed latency, {lc_int8}% INT8 usage, "
          f"{lc_sw} switches, {lc_acc}% accuracy")
    print(f"  LinUCB Warm: {lw_lat}ms stressed latency, {lw_int8}% INT8 usage, "
          f"{lw_sw} switches, {lw_acc}% accuracy")
    print(f"  LinUCB Warm starts with prior learned state instead of from scratch; "
          f"any gap above reflects the value of that warm-start knowledge")

if "EightSignalController" in results:
    es = results["EightSignalController"]
    print(f"\n  EightSignalController: {es['avg_latency_stressed']}ms stressed latency, "
          f"{es['accuracy_pct']}% overall accuracy, "
          f"{es['model_switches']} raw switches ({es['exploration_blips']} are 1-iteration "
          f"exploration blips, {es['sustained_switches']} are sustained transitions), "
          f"{es['avg_decision_time_ms']}ms avg decision time")

if "EightSignalController" in results and "LinUCB Warm" in results:
    es_lat = results["EightSignalController"]["avg_latency_stressed"]
    lw_lat = results["LinUCB Warm"]["avg_latency_stressed"]
    es_dt  = results["EightSignalController"]["avg_decision_time_ms"]
    lw_dt  = results["LinUCB Warm"]["avg_decision_time_ms"]
    es_sw  = results["EightSignalController"]["sustained_switches"]
    lw_sw  = results["LinUCB Warm"]["sustained_switches"]
    print(f"\n  EightSignalController vs LinUCB Warm — "
          f"latency: {es_lat}ms vs {lw_lat}ms, "
          f"decision time: {es_dt}ms vs {lw_dt}ms, "
          f"sustained switches: {es_sw} vs {lw_sw}")

print()