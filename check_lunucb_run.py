# analyze_linucb_run.py

import csv

rows = []
with open("linucb_run.csv", "r") as f:
    for row in csv.DictReader(f):
        rows.append(row)

n = len(rows)
fp32_rows = [r for r in rows if r["model_used"] == "fp32"]
int8_rows = [r for r in rows if r["model_used"] == "int8"]

def avg(vals):
    vals = [float(v) for v in vals if v not in (None, "", "None")]
    return sum(vals) / len(vals) if vals else float("nan")

print(f"Total rows: {n}")
print(f"FP32 chosen: {len(fp32_rows)} ({len(fp32_rows)/n:.1%})")
print(f"INT8 chosen: {len(int8_rows)} ({len(int8_rows)/n:.1%})")
print()

print(f"Avg cpu_ema_before when FP32 chosen: {avg([r['cpu_ema_before'] for r in fp32_rows]):.2f}")
print(f"Avg cpu_ema_before when INT8 chosen: {avg([r['cpu_ema_before'] for r in int8_rows]):.2f}")
print()

print(f"Avg reward when FP32 chosen: {avg([r['reward'] for r in fp32_rows]):.4f}")
print(f"Avg reward when INT8 chosen: {avg([r['reward'] for r in int8_rows]):.4f}")
print()

print(f"Avg latency_ms FP32: {avg([r['latency_ms'] for r in fp32_rows]):.2f}")
print(f"Avg latency_ms INT8: {avg([r['latency_ms'] for r in int8_rows]):.2f}")
print()

first_half = rows[:n//2]
second_half = rows[n//2:]
print(f"Avg reward — first half of run:  {avg([r['reward'] for r in first_half]):.4f}")
print(f"Avg reward — second half of run: {avg([r['reward'] for r in second_half]):.4f}")
print()

# how many times did the model actually switch, across the whole run?
switches = sum(1 for i in range(1, n) if rows[i]["model_used"] != rows[i-1]["model_used"])
print(f"Number of model switches across the run: {switches}")