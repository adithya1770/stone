import csv
import os
import statistics

TRIALS = [1, 2, 3]
DATASETS = ["imagenette", "custom"]
CONFIGS = ["baseline", "always_int8", "linucb_cold", "linucb_warm", "egreedy", "eightsignal_cold", "eightsignal_warm"]
STRESS_THRESHOLD = 70.0


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


for dataset in DATASETS:
    print("\n" + "=" * 100)
    print(f"DATASET: {dataset.upper()} — mean ± std across {len(TRIALS)} trials")
    print("=" * 100)

    header = f"{'Config':<20}{'Latency (ms)':<20}{'Accuracy (%)':<20}{'INT8 usage (%)':<20}"
    print(header)
    print("-" * 100)

    for cfg in CONFIGS:
        lat_vals, acc_vals, int8_vals = [], [], []

        for trial in TRIALS:
            path = f"results/pi_multitrial/trial{trial}/{dataset}/{cfg}_log.csv"
            if not os.path.exists(path):
                continue
            rows = read_csv(path)
            stressed = stressed_rows(rows)

            l = avg_latency(stressed)
            a = accuracy(stressed)
            i8 = int8_pct(stressed)

            if l is not None: lat_vals.append(l)
            if a is not None: acc_vals.append(a)
            if i8 is not None: int8_vals.append(i8)

        def fmt(vals):
            if not vals:
                return "n/a"
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0
            return f"{m:.1f} ± {s:.1f}"

        print(f"{cfg:<20}{fmt(lat_vals):<20}{fmt(acc_vals):<20}{fmt(int8_vals):<20}")

print("\nDone.")