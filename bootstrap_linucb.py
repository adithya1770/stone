import csv
import numpy as np
from linucb import LinUCB
from context import build_context, N_FEATURES


def bootstrap_from_csv(linucb: LinUCB, csv_path: str) -> int:
    rows_replayed = 0
    prev_latency = 0.0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("reward") or not row.get("chosen_action"):
                continue

            snapshot = {
                "cpu_ema": float(row["cpu_ema_before"]),
                "mem_ema": float(row["mem_ema_before"]),
                "temp": float(row["temp_before"]) if row["temp_before"] not in (None, "", "None") else None,
            }
            context = build_context(snapshot, prev_latency)
            action = int(row["chosen_action"])
            reward = float(row["reward"])

            linucb.update(action, context, reward)
            prev_latency = float(row["latency_ms"])
            rows_replayed += 1

    return rows_replayed


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "adaptive_run.csv"
    linucb = LinUCB(n_actions=2, n_features=N_FEATURES, alpha=1.0)
    n = bootstrap_from_csv(linucb, csv_path)
    print(f"Replayed {n} rows from {csv_path}")
    print("Learned theta (FP32 arm):", np.round(linucb.theta(0), 3))
    print("Learned theta (INT8 arm):", np.round(linucb.theta(1), 3))