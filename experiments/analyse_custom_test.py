import csv
from collections import Counter

IN_FILE = "logging/custom_test_results.csv"

rows = []
with open(IN_FILE) as f:
    for row in csv.DictReader(f):
        rows.append(row)

algos = sorted(set(r["algo"] for r in rows))

print("\n" + "=" * 60)
print("MODEL CHOICE UNDER SUSTAINED CPU STRESS (cpu ~100%)")
print("=" * 60)

for algo in algos:
    subset = [r for r in rows if r["algo"] == algo]
    counts = Counter(r["model"] for r in subset)
    total = len(subset)
    int8_pct = round(100 * counts.get("int8", 0) / total, 1) if total else 0
    print(f"\n{algo}")
    print(f"  Total images: {total}")
    print(f"  FP32 chosen:  {counts.get('fp32', 0)}")
    print(f"  INT8 chosen:  {counts.get('int8', 0)}")
    print(f"  INT8 usage under stress: {int8_pct}%")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("Under sustained ~100% CPU stress, an adaptive algorithm should")
print("favor INT8 (faster, lighter) over FP32. A 0% INT8 rate despite")
print("full CPU stress indicates the algorithm failed to react to")
print("device conditions.")
print("=" * 60 + "\n")