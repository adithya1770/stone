import time
import csv
import os
import subprocess
import itertools
from telemetry import Telemetry
from model_manager import ModelManager
from decision_engine import DecisionEngine
from policies import FixedPolicy, HeuristicPolicy, LinUCBPolicy
from linucb import LinUCB
from context import N_FEATURES
from bootstrap_linucb import bootstrap_from_csv

# Reporting bucket cutoff -- deliberately independent of heuristics.py's own
# CPU_THRESHOLD (30). Answers "how did we do under genuinely heavy load"
# using a fixed definition, regardless of the heuristic's internal switch point.
STRESSED_THRESHOLD = 70

TEST_IMAGES_DIR = "test_images"
ACTION_TO_MODEL_TYPE = {0: "fp32", 1: "int8"}
DURATION_SECONDS = 300   # 5 minutes per condition, identical for all four
LOOP_INTERVAL = 3

# (offset_seconds, duration_seconds) -- same schedule fired for every condition
STRESS_SCHEDULE = [(60, 120)]  # idle 0-60s, stress 60-180s, idle 180-300s


def load_ground_truth(csv_path):
    gt = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            gt[row["filename"]] = row["ground_truth_label"]
    return gt


def compute_reward(predicted_label, ground_truth_label, latency_ms):
    correct = 1.0 if predicted_label == ground_truth_label else 0.0
    return correct - (0.01 * latency_ms)


def run_condition(name, policy, mm, ground_truth, log_path):
    print(f"\n--- Running condition: {name} ({DURATION_SECONDS}s) ---")
    telemetry = Telemetry()
    engine = DecisionEngine(policy, min_mode_duration=5)
    image_cycle = itertools.cycle(sorted(ground_truth.keys()))

    rows = []
    start = time.time()
    input_id = 0
    prev_model = None
    switches = 0
    stress_launched = [False] * len(STRESS_SCHEDULE)

    with open(log_path, "w", newline="") as f:
        fieldnames = ["input_id", "cpu_ema", "model_used", "confidence", "latency_ms", "correct", "reward"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while time.time() - start < DURATION_SECONDS:
            elapsed = time.time() - start
            for i, (offset, dur) in enumerate(STRESS_SCHEDULE):
                if not stress_launched[i] and elapsed >= offset:
                    subprocess.Popen(["stress-ng", "--cpu", "8", "--timeout", f"{dur}s"])
                    stress_launched[i] = True
                    print(f"  [t={elapsed:.0f}s] stress-ng launched for {dur}s")

            snap = telemetry.read()
            action, context = engine.decide(snap)
            model_type = ACTION_TO_MODEL_TYPE[action]

            image_filename = next(image_cycle)
            image_path = os.path.join(TEST_IMAGES_DIR, image_filename)
            gt_label = ground_truth[image_filename]

            result = mm.predict(image_path, model_type)
            reward = compute_reward(result["label"], gt_label, result["latency_ms"])
            engine.learn(action, context, reward, result["latency_ms"])

            if prev_model is not None and model_type != prev_model:
                switches += 1
            prev_model = model_type

            row = {
                "input_id": input_id, "cpu_ema": snap["cpu_ema"], "model_used": model_type,
                "confidence": result["confidence"], "latency_ms": result["latency_ms"],
                "correct": int(result["label"] == gt_label), "reward": reward,
            }
            writer.writerow(row)
            rows.append(row)

            input_id += 1
            time.sleep(LOOP_INTERVAL)

    return rows, switches


def summarize(name, rows, switches):
    normal = [r for r in rows if r["cpu_ema"] < STRESSED_THRESHOLD]
    stressed = [r for r in rows if r["cpu_ema"] >= STRESSED_THRESHOLD]

    def avg(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    int8_under_stress = sum(1 for r in stressed if r["model_used"] == "int8")
    int8_pct = (int8_under_stress / len(stressed) * 100) if stressed else 0.0

    return {
        "name": name, "total": len(rows), "n_normal": len(normal), "n_stressed": len(stressed),
        "avg_lat_normal": avg([r["latency_ms"] for r in normal]),
        "avg_lat_stressed": avg([r["latency_ms"] for r in stressed]),
        "max_lat_stressed": max([r["latency_ms"] for r in stressed], default=float("nan")),
        "avg_conf_normal": avg([r["confidence"] for r in normal]),
        "avg_conf_stressed": avg([r["confidence"] for r in stressed]),
        "switches": switches, "int8_pct_stressed": int8_pct,
    }


def print_table(conditions):
    print("\n" + "=" * 95)
    print("EXPERIMENT RESULTS")
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


if __name__ == "__main__":
    ground_truth = load_ground_truth(os.path.join(TEST_IMAGES_DIR, "ground_truth.csv"))
    mm = ModelManager("models/mobilenet_v2_fp32.tflite", "models/mobilenet_v2_int8.tflite", "models/labels.txt")

    conditions = []

    rows, sw = run_condition("Baseline", FixedPolicy(action=0), mm, ground_truth, "exp_baseline.csv")
    conditions.append(summarize("Baseline", rows, sw))

    rows, sw = run_condition("Rule-Based", HeuristicPolicy(), mm, ground_truth, "exp_rulebased.csv")
    conditions.append(summarize("Rule-Based", rows, sw))

    cold_bandit = LinUCB(n_actions=2, n_features=N_FEATURES, alpha=1.0)
    rows, sw = run_condition("LinUCB Cold", LinUCBPolicy(cold_bandit), mm, ground_truth, "exp_linucb_cold.csv")
    conditions.append(summarize("LinUCB Cold", rows, sw))

    warm_bandit = LinUCB(n_actions=2, n_features=N_FEATURES, alpha=1.0)
    n_replayed = bootstrap_from_csv(warm_bandit, "adaptive_run.csv")
    print(f"\nBootstrapped warm LinUCB with {n_replayed} historical rows")
    rows, sw = run_condition("LinUCB Warm", LinUCBPolicy(warm_bandit), mm, ground_truth, "exp_linucb_warm.csv")
    conditions.append(summarize("LinUCB Warm", rows, sw))

    print_table(conditions)