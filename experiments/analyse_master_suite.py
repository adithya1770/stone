import csv
import os
import statistics
import matplotlib.pyplot as plt

DATASETS = ["imagenette", "custom"]
CONFIGS = ["baseline", "always_int8", "linucb_cold", "linucb_warm", "egreedy", "eightsignal_cold", "eightsignal_warm"]
LABELS = ["Baseline", "AlwaysINT8", "LinUCB\nCold", "LinUCB\nWarm", "EpsilonGreedy", "EightSignal\nCold", "EightSignal\nWarm"]
STRESS_THRESHOLD = 70.0


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def stressed_rows(rows):
    return [r for r in rows if float(r["cpu"]) >= STRESS_THRESHOLD]


def avg(vals):
    return round(statistics.mean(vals), 2) if vals else 0.0


def accuracy(rows):
    known = [r for r in rows if r["correct"] in ("True", "False")]
    return round(100 * sum(1 for r in known if r["correct"] == "True") / len(known), 2) if known else None


def int8_pct(rows):
    return round(100 * sum(1 for r in rows if r["model"] == "int8") / len(rows), 1) if rows else 0.0


for dataset in DATASETS:
    in_dir = f"results/master_suite/{dataset}"
    out_dir = f"{in_dir}/plots"
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 120)
    print(f"DATASET: {dataset.upper()}")
    print("=" * 120)

    latency_vals, acc_vals, int8_vals = [], [], []

    header = f"{'Metric':<28}" + "".join(f"{c:>14}" for c in CONFIGS)
    print(header)
    print("-" * 120)

    metric_rows = {"latency": [], "accuracy": [], "switches": [], "int8_pct": []}

    for cfg in CONFIGS:
        path = os.path.join(in_dir, f"{cfg}_log.csv")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            metric_rows["latency"].append(0)
            metric_rows["accuracy"].append(0)
            metric_rows["switches"].append(0)
            metric_rows["int8_pct"].append(0)
            continue

        rows = read_csv(path)
        stressed = stressed_rows(rows)

        lat = avg([float(r["latency_ms"]) for r in stressed])
        acc = accuracy(stressed)
        switches = sum(1 for i in range(1, len(rows)) if rows[i]["model"] != rows[i-1]["model"])
        i8 = int8_pct(stressed)

        metric_rows["latency"].append(lat)
        metric_rows["accuracy"].append(acc if acc is not None else 0)
        metric_rows["switches"].append(switches)
        metric_rows["int8_pct"].append(i8)

    for label, key in [("Avg latency stressed (ms)", "latency"),
                        ("Accuracy stressed (%)", "accuracy"),
                        ("Model switches", "switches"),
                        ("INT8 usage stressed (%)", "int8_pct")]:
        row = f"{label:<28}"
        for v in metric_rows[key]:
            row += f"{v:>14}"
        print(row)

    # Plots
    plt.figure(figsize=(10, 5))
    plt.bar(LABELS, metric_rows["latency"], color="#4a7fbf")
    plt.ylabel("Avg Latency Under Stress (ms)")
    plt.title(f"Stressed Latency — {dataset}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/stressed_latency.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(LABELS, metric_rows["accuracy"], color="#2ecc71")
    plt.ylabel("Accuracy Under Stress (%)")
    plt.title(f"Stressed Accuracy — {dataset}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/stressed_accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(LABELS, metric_rows["int8_pct"], color="#e67e22")
    plt.ylabel("INT8 Usage Under Stress (%)")
    plt.title(f"Adaptive Response to Stress — {dataset}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/int8_under_stress.png", dpi=150)
    plt.close()

    print(f"\nPlots saved to {out_dir}/")

print("\n" + "=" * 120)
print("Done.")