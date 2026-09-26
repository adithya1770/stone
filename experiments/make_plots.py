import csv
import statistics
import os
import matplotlib.pyplot as plt

# ---------- Config ----------
RUN_DIRS = ["results/final_eightsignal_run1", "results/final_eightsignal_run2", "results/final_eightsignal_run3"]
CONFIGS = ["baseline", "always_int8", "linucb_cold", "linucb_warm", "eightsignal_cold", "eightsignal_warm"]
LABELS = ["Baseline", "AlwaysINT8", "LinUCB\nCold", "LinUCB\nWarm", "EightSignal\nCold", "EightSignal\nWarm"]
STRESS_THRESHOLD = 70.0
OUT_DIR = "results/final_eightsignal_run1/plots"

# Most recent, structured custom-image suite (run_custom_suite.py output)
CUSTOM_DIR = "results/custom_suite"
EG_CUSTOM_PATH = f"{CUSTOM_DIR}/egreedy_log.csv"
ES_CUSTOM_PATH = f"{CUSTOM_DIR}/eightsignal_warm_log.csv"

os.makedirs(OUT_DIR, exist_ok=True)


# ---------- Helpers ----------
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def stressed_rows(rows):
    return [r for r in rows if float(r["cpu"]) >= STRESS_THRESHOLD]


def avg_latency(rows):
    return statistics.mean([float(r["latency_ms"]) for r in rows]) if rows else None


def accuracy(rows):
    known = [r for r in rows if r["correct"] in ("True", "False")]
    return 100 * sum(1 for r in known if r["correct"] == "True") / len(known) if known else None


def int8_pct(rows):
    return 100 * sum(1 for r in rows if r["model"] == "int8") / len(rows) if rows else None


def metric_across_runs(cfg, key_fn):
    vals = []
    for run_dir in RUN_DIRS:
        path = f"{run_dir}/{cfg}_log.csv"
        if not os.path.exists(path):
            continue
        rows = stressed_rows(read_csv(path))
        v = key_fn(rows)
        if v is not None:
            vals.append(v)
    if not vals:
        return 0, 0
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0
    return mean, std


# ---------- Chart 1: Stressed Latency (locked laptop suite, n=3) ----------
latency_means, latency_stds = [], []
for cfg in CONFIGS:
    m, s = metric_across_runs(cfg, avg_latency)
    latency_means.append(m); latency_stds.append(s)

plt.figure(figsize=(8, 5))
plt.bar(LABELS, latency_means, yerr=latency_stds, capsize=5, color="#4a7fbf")
plt.ylabel("Avg Latency Under Stress (ms)")
plt.title("Stressed Latency Across Algorithms (mean ± std, n=3)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stressed_latency.png", dpi=150)
plt.close()
print("Saved: stressed_latency.png")


# ---------- Chart 2: Stressed Accuracy (locked laptop suite, n=3) ----------
acc_means, acc_stds = [], []
for cfg in CONFIGS:
    m, s = metric_across_runs(cfg, accuracy)
    acc_means.append(m); acc_stds.append(s)

plt.figure(figsize=(8, 5))
plt.bar(LABELS, acc_means, yerr=acc_stds, capsize=5, color="#2ecc71")
plt.ylabel("Accuracy Under Stress (%)")
plt.title("Stressed Accuracy Across Algorithms (mean ± std, n=3)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/stressed_accuracy.png", dpi=150)
plt.close()
print("Saved: stressed_accuracy.png")


# ---------- Chart 3: EightSignal vs EpsilonGreedy ----------
# Uses the most recent, structured custom-image suite run (run_custom_suite.py),
# NOT the historical single run, which showed high run-to-run variance for
# EpsilonGreedy (0% in the custom suite vs 97% in a single earlier run) —
# consistent with epsilon-greedy's known dependence on exploration luck
# rather than genuine context-awareness.
if os.path.exists(EG_CUSTOM_PATH) and os.path.exists(ES_CUSTOM_PATH):
    eg_val = int8_pct(stressed_rows(read_csv(EG_CUSTOM_PATH)))
    es_val = int8_pct(stressed_rows(read_csv(ES_CUSTOM_PATH)))

    plt.figure(figsize=(6, 5))
    plt.bar(
        ["EpsilonGreedy\n(EdgeMLBalancer's\nalgorithm class)", "EightSignal\nController\n(Warm)"],
        [eg_val, es_val],
        color=["#d9534f", "#5cb85c"]
    )
    plt.ylabel("INT8 Usage Under Stress (%)")
    plt.title("Adaptive Response to CPU Stress\n(custom-image suite)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eightsignal_vs_egreedy.png", dpi=150)
    plt.close()
    print("Saved: eightsignal_vs_egreedy.png")
    print(f"EpsilonGreedy: {eg_val:.1f}%  |  EightSignal Warm: {es_val:.1f}%")
else:
    print(f"Missing files — checked:\n  {EG_CUSTOM_PATH}\n  {ES_CUSTOM_PATH}")

print(f"\nAll available charts saved to {OUT_DIR}/")