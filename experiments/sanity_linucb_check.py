"""
Sanity-check simulation for LinUCB, using SYNTHETIC reward data (not real
model inference) — purely to verify the algorithm's mechanics work correctly.

Made-up ground truth: FP32 reward degrades as CPU rises (like real latency
penalty under contention), INT8 stays relatively flatter. They cross around
cpu ~= 50 — so there IS a genuinely correct action per context, same as the
real project's FP32-vs-INT8 tradeoff.
"""

import numpy as np
from linucb import LinUCB
from context import build_context, N_FEATURES

rng = np.random.default_rng(42)

def true_reward(action, cpu):
    noise = rng.normal(0, 0.05)
    if action == 0:  # FP32
        return (0.9 - 0.005 * cpu) + noise
    else:  # INT8
        return (0.5 + 0.003 * cpu) + noise

def optimal_action(cpu):
    r0 = 0.9 - 0.005 * cpu
    r1 = 0.5 + 0.003 * cpu
    return 0 if r0 > r1 else 1

linucb = LinUCB(n_actions=2, n_features=N_FEATURES, alpha=1.0)

N_ROUNDS = 2000
correct_matches = []
prev_latency = 10.0

for t in range(N_ROUNDS):
    cpu = rng.uniform(0, 100)
    mem = rng.uniform(30, 60)
    temp = rng.uniform(40, 70)
    snapshot = {"cpu_ema": cpu, "mem_ema": mem, "temp": temp}
    context = build_context(snapshot, prev_latency)

    action = linucb.select_action(context)
    reward = true_reward(action, cpu)
    linucb.update(action, context, reward)

    correct_matches.append(1 if action == optimal_action(cpu) else 0)
    prev_latency = rng.uniform(5, 30)

correct_matches = np.array(correct_matches)
print(f"Match rate — first 100 rounds: {correct_matches[:100].mean():.2%}")
print(f"Match rate — last 100 rounds:  {correct_matches[-100:].mean():.2%}")
print(f"Match rate — overall:          {correct_matches.mean():.2%}")
print()
print("Learned theta (FP32 arm):", np.round(linucb.theta(0), 3))
print("Learned theta (INT8 arm):", np.round(linucb.theta(1), 3))
print()

low_cpu_ctx = build_context({"cpu_ema": 10, "mem_ema": 40, "temp": 50}, 10)
high_cpu_ctx = build_context({"cpu_ema": 90, "mem_ema": 40, "temp": 50}, 10)
print(f"Chosen action at LOW cpu (10%):  {linucb.select_action(low_cpu_ctx)} (expect 0 = FP32)")
print(f"Chosen action at HIGH cpu (90%): {linucb.select_action(high_cpu_ctx)} (expect 1 = INT8)")