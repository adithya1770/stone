import csv
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.reward import calculate_reward

FILES = {
    "Baseline":          "logging/baseline_log.csv",
    "AlwaysINT8":        "logging/always_int8_log.csv",
    "Rule-Based":        "logging/rule_based_log.csv",
    "EpsilonGreedy":     "logging/egreedy_log.csv",
    "LinUCB Cold":       "logging/linucb_log.csv",
    "LinUCB Warm":       "logging/linucb_warm_log.csv",
    "Thompson":          "logging/thompson_log.csv",
    "SlidingLinUCBWarm": "logging/sliding_log_warm.csv",
    "SlidingLinUCBCold": "logging/sliding_log_cold.csv",
}

OUTPUT_DIR = "logging/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "cpu":         float(row["cpu"]),
                "ram":         float(row["ram"]),
                "temperature": float(row["temperature"]),
                "latency_ms":  float(row["latency_ms"]),
                "confidence":  float(row["confidence"]),
                "model":       row["model"],
                "decision_time_ms": float(row.get("decision_time_ms", 0) or 0)
            })
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


def plot_single_system(name, rows):
    """One figure per system: latency, model choice, reward, CPU — all over
    inference index, stacked vertically so you can see how they relate."""

    x = list(range(len(rows)))
    latency = [r["latency_ms"] for r in rows]
    reward = [r["reward"] for r in rows]
    cpu = [r["cpu"] for r in rows]
    model_numeric = [0 if r["model"] == "fp32" else 1 for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(f"{name} — behaviour over the run", fontsize=13)

    axes[0].plot(x, cpu, color="tab:red")
    axes[0].axhline(70, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("CPU %")
    axes[0].set_title("Telemetry (CPU) — dashed line marks stress threshold (70%)")

    axes[1].step(x, model_numeric, where="post", color="tab:purple")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["fp32", "int8"])
    axes[1].set_title("Model chosen at each step")

    axes[2].plot(x, latency, color="tab:blue")
    axes[2].set_ylabel("Latency (ms)")
    axes[2].set_title("Inference latency")

    axes[3].plot(x, reward, color="tab:green")
    axes[3].set_ylabel("Reward")
    axes[3].set_xlabel("Inference index")
    axes[3].set_title("Reward (recomputed from cpu/ram/temp/confidence/latency)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(OUTPUT_DIR, f"{name.replace(' ', '_')}_timeline.png")
    plt.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_comparison_bars(all_results):
    """One figure comparing all systems side by side on the metrics that
    matter most for the report: stressed latency, switches, avg reward."""

    names = list(all_results.keys())
    stressed_latency = [all_results[n]["avg_latency_stressed"] for n in names]
    switches = [all_results[n]["model_switches"] for n in names]
    avg_reward = [all_results[n]["avg_reward"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].barh(names, stressed_latency, color="tab:blue")
    axes[0].set_xlabel("ms")
    axes[0].set_title("Avg latency — stressed")
    axes[0].invert_yaxis()

    axes[1].barh(names, switches, color="tab:orange")
    axes[1].set_xlabel("count")
    axes[1].set_title("Model switches")
    axes[1].invert_yaxis()

    axes[2].barh(names, avg_reward, color="tab:green")
    axes[2].set_xlabel("reward")
    axes[2].set_title("Avg reward (whole run)")
    axes[2].invert_yaxis()

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "comparison_summary.png")
    plt.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"Saved: {outpath}")


def summarize(rows):
    stressed = [r for r in rows if r["cpu"] >= 70.0]
    switches = sum(
        1 for i in range(1, len(rows))
        if rows[i]["model"] != rows[i-1]["model"]
    )
    return {
        "avg_latency_stressed": (
            sum(r["latency_ms"] for r in stressed) / len(stressed)
            if stressed else 0
        ),
        "model_switches": switches,
        "avg_reward": sum(r["reward"] for r in rows) / len(rows) if rows else 0
    }


all_results = {}

for name, path in FILES.items():
    if not os.path.exists(path):
        print(f"Missing: {path} — skipped")
        continue

    rows = read_csv(path)
    rows = add_reward(rows)

    plot_single_system(name, rows)
    all_results[name] = summarize(rows)

if all_results:
    plot_comparison_bars(all_results)

print(f"\nAll plots saved to: {OUTPUT_DIR}/")