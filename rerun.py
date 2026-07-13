# reanalyze.py

import csv

STRESSED_THRESHOLD = 40  # <-- update this once you see the real min/max/avg output above

files = {
    "Baseline": "exp_baseline.csv",
    "Rule-Based": "exp_rulebased.csv",
    "LinUCB Cold": "exp_linucb_cold.csv",
    "LinUCB Warm": "exp_linucb_warm.csv",
}

def summarize(name, rows):
    normal = [r for r in rows if float(r["cpu_ema"]) < STRESSED_THRESHOLD]
    stressed = [r for r in rows if float(r["cpu_ema"]) >= STRESSED_THRESHOLD]

    def avg(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    switches = sum(1 for i in range(1, len(rows)) if rows[i]["model_used"] != rows[i-1]["model_used"])
    int8_stressed = sum(1 for r in stressed if r["model_used"] == "int8")
    int8_pct = (int8_stressed / len(stressed) * 100) if stressed else 0.0

    return {
        "name": name, "total": len(rows), "n_normal": len(normal), "n_stressed": len(stressed),
        "avg_lat_normal": avg([float(r["latency_ms"]) for r in normal]),
        "avg_lat_stressed": avg([float(r["latency_ms"]) for r in stressed]),
        "max_lat_stressed": max([float(r["latency_ms"]) for r in stressed], default=float("nan")),
        "avg_conf_normal": avg([float(r["confidence"]) for r in normal]),
        "avg_conf_stressed": avg([float(r["confidence"]) for r in stressed]),
        "switches": switches, "int8_pct_stressed": int8_pct,
    }

conditions = []
for name, path in files.items():
    with open(path) as f:
        rows = list(csv.DictReader(f))
    conditions.append(summarize(name, rows))

print("=" * 95)
header = f"{'Metric':<28}" + "".join(f"{c['name']:>16}" for c in conditions)
print(header)
print("-" * 95)
def row(label, key, fmt="{}"):
    print(f"{label:<28}" + "".join(f"{fmt.format(c[key]):>16}" for c in conditions))
row("Total readings", "total")
row("Normal readings", "n_normal")
row("Stressed readings", "n_stressed")
row("Avg latency - normal", "avg_lat_normal", "{:.2f}ms")
row("Avg latency - stressed", "avg_lat_stressed", "{:.2f}ms")
row("Max latency - stressed", "max_lat_stressed", "{:.2f}ms")
row("Avg confidence - normal", "avg_conf_normal", "{:.2f}")
row("Avg confidence - stressed", "avg_conf_stressed", "{:.2f}")
row("Model switches", "switches")
row("INT8 usage under stress", "int8_pct_stressed", "{:.1f}%")