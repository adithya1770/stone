"""Turn the raw Stone multi-trial logs into tidy tables for analysis and Power BI.
"""
import csv
import os
import statistics as st
import sys

RAW_DIR = sys.argv[1] if len(sys.argv) > 1 else "../stone-results/results/stone_pi_multitrial"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
STRESS_CPU = 70.0
TRIALS = [1, 2, 3]
DATASETS = {"imagenette": "ImageNette", "custom": "Custom"}

# config, display name, policy family, type, start mode
CONFIGS = [
    ("baseline", "Baseline (FP32)", "Static baseline", "Static", "n/a"),
    ("always_int8", "Always INT8", "Static baseline", "Static", "n/a"),
    ("egreedy", "ε-greedy", "ε-greedy", "Adaptive", "cold"),
    ("linucb_cold", "LinUCB (cold)", "LinUCB", "Adaptive", "cold"),
    ("linucb_warm", "LinUCB (warm)", "LinUCB", "Adaptive", "warm"),
    ("eightsignal_cold", "EightSignal (cold)", "EightSignal", "Adaptive", "cold"),
    ("eightsignal_warm", "EightSignal (warm)", "EightSignal", "Adaptive", "warm"),
]


def accuracy(rows):
    known = [r for r in rows if r["correct"] in ("True", "False")]
    if not known:
        return ""
    return round(100 * sum(r["correct"] == "True" for r in known) / len(known), 2)


def mean(values):
    return round(st.mean(values), 2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(OUT_DIR, "configs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "config_name", "policy_family", "type", "start_mode", "sort_order"])
        for i, row in enumerate(CONFIGS, start=1):
            w.writerow([*row, i])

    steps, summary = [], []
    for ds_key, ds_name in DATASETS.items():
        for cfg, *_ in CONFIGS:
            for trial in TRIALS:
                path = os.path.join(RAW_DIR, f"trial{trial}", ds_key, f"{cfg}_log.csv")
                with open(path, encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))

                for i, r in enumerate(rows):
                    steps.append({
                        "run_id": f"{ds_name}|{cfg}|{trial}",
                        "dataset": ds_name,
                        "config": cfg,
                        "trial": trial,
                        "step": int(r["iteration"]),
                        "cpu_pct": r["cpu"],
                        "ram_pct": r["ram"],
                        "temperature_c": r["temperature"],
                        "model": r["model"].upper(),
                        "is_int8": int(r["model"] == "int8"),
                        "load_state": "Stressed" if float(r["cpu"]) >= STRESS_CPU else "Normal",
                        "true_label": r["true_label"],
                        "predicted": r["predicted"],
                        "is_correct": {"True": 1, "False": 0}.get(r["correct"], ""),
                        "confidence": r["confidence"],
                        "latency_ms": r["latency_ms"],
                        "decision_time_ms": r["decision_time_ms"],
                        "switched": int(i > 0 and r["model"] != rows[i - 1]["model"]),
                    })

                stressed = [r for r in rows if float(r["cpu"]) >= STRESS_CPU]
                summary.append({
                    "dataset": ds_name,
                    "config": cfg,
                    "trial": trial,
                    "stressed_latency_ms": mean(float(r["latency_ms"]) for r in stressed),
                    "stressed_accuracy_pct": accuracy(stressed),
                    "stressed_int8_pct": round(100 * sum(r["model"] == "int8" for r in stressed) / len(stressed), 2),
                    "overall_latency_ms": mean(float(r["latency_ms"]) for r in rows),
                    "overall_accuracy_pct": accuracy(rows),
                    "overall_int8_pct": round(100 * sum(r["model"] == "int8" for r in rows) / len(rows), 2),
                    "model_switches": sum(1 for i in range(1, len(rows)) if rows[i]["model"] != rows[i - 1]["model"]),
                    "avg_decision_time_ms": round(st.mean(float(r["decision_time_ms"]) for r in rows), 3),
                })

    for name, data in [("steps.csv", steps), ("trial_summary.csv", summary)]:
        with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    # Print the mean +/- SD table so it can be checked against the project's result_final.txt
    for ds_name in DATASETS.values():
        print(f"\n{ds_name} (stressed window, mean ± SD over {len(TRIALS)} trials)")
        print(f"{'config':<18}{'latency ms':>16}{'accuracy %':>16}{'INT8 %':>16}")
        for cfg, *_ in CONFIGS:
            runs = [s for s in summary if s["dataset"] == ds_name and s["config"] == cfg]
            cells = []
            for key in ("stressed_latency_ms", "stressed_accuracy_pct", "stressed_int8_pct"):
                vals = [r[key] for r in runs]
                cells.append(f"{st.mean(vals):.1f} ± {st.stdev(vals):.1f}")
            print(f"{cfg:<18}" + "".join(f"{c:>16}" for c in cells))

    print(f"\nWrote {len(steps)} step rows and {len(summary)} run rows to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
