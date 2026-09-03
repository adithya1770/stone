import time
import sys
import os
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.telemetry import get_telemetry
from runtime.decision_engine import get_algo

ITERATIONS = 100
ALGOS_TO_TEST = ["boolean", "disagreement", "andgate"]


def run_stress_workload():
    proc = subprocess.Popen(
        ["bash", "-c",
         "sleep 25 && stress-ng --cpu 0 --timeout 40s && sleep 25"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return proc


for algo_name in ALGOS_TO_TEST:
    print("\n" + "=" * 60)
    print(f"Testing: {algo_name}")
    print("=" * 60)

    algo = get_algo(algo_name, alpha=1.0)

    source_counts = {}

    workload = run_stress_workload()

    for i in range(ITERATIONS):
        t = get_telemetry()
        chosen, scores = algo.choose(
            t["cpu"], t["ram"], t["temperature"], t["battery"]
        )

        source = algo.get_last_source() if hasattr(algo, "get_last_source") else "n/a"
        source_counts[source] = source_counts.get(source, 0) + 1

        reward = 0.5
        algo.update(chosen, t["cpu"], t["ram"], t["temperature"], t["battery"], reward)

        stress_tag = "STRESSED" if t["cpu"] >= 70 else "normal"
        print(f"  [{i+1:3d}/{ITERATIONS}] cpu={t['cpu']:5.1f}  "
              f"chosen={chosen:5s}  source={source:16s}  {stress_tag}")

        time.sleep(1)

    workload.wait()

    print(f"\n  Source breakdown for {algo_name}:")
    for source, count in source_counts.items():
        pct = round(100 * count / ITERATIONS, 1)
        print(f"    {source:20s}: {count:3d} / {ITERATIONS}  ({pct}%)")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS COMPLETE")
print("=" * 60)