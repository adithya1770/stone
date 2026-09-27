import csv
import os
import statistics
import matplotlib.pyplot as plt

TRIALS = [1, 2, 3]
DATASETS = ["imagenette", "custom"]
CONFIGS = ["baseline", "always_int8", "linucb_cold", "linucb_warm", "egreedy", "eightsignal_cold", "eightsignal_warm"]
LABELS = ["Baseline", "AlwaysINT8", "LinUCB\nCold", "LinUCB\nWarm", "EpsilonGreedy", "EightSignal\nCold", "EightSignal\nWarm"]
STRESS_THRESHOLD = 70.0


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


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


def analyse_single_run(rows):
    normal = [r for r in rows if float(r["cpu"]) < STRESS_THRESHOLD]
    stressed = [r for r in rows if float(r["cpu"]) >= STRESS_THRESHOLD]

    def avg(vals):
        return sum(vals) / len(vals) if vals else 0

    def avg_latency(subset):
        return avg([float(r["latency_ms"]) for r in subset])

    def avg_confidence(subset):
        return avg([float(r["confidence"]) for r in subset])

    def accuracy_pct(subset):
        known = [r for r in subset if r["correct"] in ("True", "False")]
        if not known:
            return None
        return 100 * sum(1 for r in known if r["correct"] == "True") / len(known)

    switches = sum(1 for i in range(1, len(rows)) if rows[i]["model"] != rows[i-1]["model"])
    run_lengths = get_run_lengths(rows)
    blip_runs = [l for l in run_lengths if l == 1]
    sustained_runs = [l for l in run_lengths if l >= 2]
    exploration_blips = len(blip_runs)
    sustained_switches = max(len(sustained_runs) - 1, 0)

    int8_count = sum(1 for r in stressed if r["model"] == "int8")
    int8_pct = 100 * int8_count / len(stressed) if stressed else 0

    return {
        "avg_latency_normal": avg_latency(normal),
        "avg_latency_stressed": avg_latency(stressed),
        "max_latency_stressed": max([float(r["latency_ms"]) for r in stressed], default=0),
        "avg_confidence_normal": avg_confidence(normal),
        "avg_confidence_stressed": avg_confidence(stressed),
        "accuracy_pct": accuracy_pct(rows),
        "accuracy_pct_normal": accuracy_pct(normal),
        "accuracy_pct_stressed": accuracy_pct(stressed),
        "model_switches": switches,
        "exploration_blips": exploration_blips,
        "sustained_switches": sustained_switches,
        "int8_under_stress_pct": int8_pct,
        "avg_decision_time_ms": avg([float(r["decision_time_ms"]) for r in rows]),
    }


ALL_METRIC_KEYS = [
    ("Avg latency — normal (ms)", "avg_latency_normal"),
    ("Avg latency — stressed (ms)", "avg_latency_stressed"),
    ("Max latency — stressed (ms)", "max_latency_stressed"),
    ("Avg confidence — normal", "avg_confidence_normal"),
    ("Avg confidence — stressed", "avg_confidence_stressed"),
    ("Accuracy — overall (%)", "accuracy_pct"),
    ("Accuracy — normal (%)", "accuracy_pct_normal"),
    ("Accuracy — stressed (%)", "accuracy_pct_stressed"),
    ("Model switches (raw)", "model_switches"),
    ("  exploration blips", "exploration_blips"),
    ("Sustained switches", "sustained_switches"),
    ("INT8 usage under stress (%)", "int8_under_stress_pct"),
    ("Avg decision time (ms)", "avg_decision_time_ms"),
]

for dataset in DATASETS:
    out_dir = f"results/pi_multitrial/plots/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 130)
    print(f"DATASET: {dataset.upper()} — mean ± std across {len(TRIALS)} trials")
    print("=" * 130)

    per_config_trials = {cfg: [] for cfg in CONFIGS}

    for cfg in CONFIGS:
        for trial in TRIALS:
            path = f"results/pi_multitrial/trial{trial}/{dataset}/{cfg}_log.csv"
            if not os.path.exists(path):
                continue
            rows = read_csv(path)
            per_config_trials[cfg].append(analyse_single_run(rows))

    header = f"{'Metric':<32}" + "".join(f"{c:>16}" for c in CONFIGS)
    print(header)
    print("-" * 130)

    plot_data = {key: {"mean": [], "std": []} for _, key in ALL_METRIC_KEYS}

    for label, key in ALL_METRIC_KEYS:
        row_str = f"{label:<32}"
        for cfg in CONFIGS:
            vals = [t[key] for t in per_config_trials[cfg] if t[key] is not None]
            if vals:
                m = statistics.mean(vals)
                s = statistics.stdev(vals) if len(vals) > 1 else 0
            else:
                m, s = 0, 0
            row_str += f"{m:>9.1f}±{s:<5.1f}"
            plot_data[key]["mean"].append(m)
            plot_data[key]["std"].append(s)
        print(row_str)

    print("=" * 130)

    def make_plot(key, ylabel, title, color, filename):
        plt.figure(figsize=(9, 5))
        plt.bar(LABELS, plot_data[key]["mean"], yerr=plot_data[key]["std"], capsize=5, color=color)
        plt.ylabel(ylabel)
        plt.title(f"{title} — {dataset} (mean ± std, n={len(TRIALS)})")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{filename}.png", dpi=150)
        plt.close()

    make_plot("avg_latency_stressed", "Avg Latency Under Stress (ms)", "Stressed Latency", "#4a7fbf", "stressed_latency")
    make_plot("accuracy_pct_stressed", "Accuracy Under Stress (%)", "Stressed Accuracy", "#2ecc71", "stressed_accuracy")
    make_plot("int8_under_stress_pct", "INT8 Usage Under Stress (%)", "Adaptive Response to Stress", "#e67e22", "int8_under_stress")
    make_plot("sustained_switches", "Sustained Switches", "Model Switch Stability", "#9b59b6", "sustained_switches")
    make_plot("avg_confidence_stressed", "Avg Confidence Under Stress", "Confidence Under Stress", "#1abc9c", "confidence_stressed")

    print(f"\nPlots saved to {out_dir}/")

print("\nDone.")