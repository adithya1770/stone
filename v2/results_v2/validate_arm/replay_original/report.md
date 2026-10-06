# STONE v2 replay report (tiny, original preprocessing)

Table: `v2/results_v2/validate_arm/table_tiny.csv` — 3000 images (1494 tune / 1506 test, split by hash).

## How well does the cheap model's score flag its own mistakes? (AUROC, test split)

0.5 = no better than chance, 1.0 = perfect. High AUROC is what makes escalation pay off.

| variant | accuracy % | AUROC margin | AUROC confidence | AUROC -entropy |
|---|---|---|---|---|
| int8_96 | 6.4 | 0.554 | 0.660 | 0.701 |
| int8_128 | 12.7 | 0.525 | 0.616 | 0.663 |
| int8_160 | 14.9 | 0.613 | 0.666 | 0.691 |
| int8_192 | 13.8 | 0.697 | 0.717 | 0.726 |
| int8_224 | 10.8 | 0.695 | 0.715 | 0.728 |
| fp32_128 | 16.3 | 0.731 | 0.758 | 0.775 |
| fp32_160 | 16.9 | 0.704 | 0.717 | 0.730 |
| fp32_224 | 10.4 | 0.671 | 0.678 | 0.660 |

## Condition: idle

Latency source: ['profile']; budget = 2.87 ms (AlwaysINT8 mean latency)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static int8_96 | 6.4 | 0.59 | 0.65 | – |
| static int8_128 | 12.7 | 0.97 | 1.09 | – |
| static int8_160 | 14.9 | 1.51 | 1.68 | – |
| static int8_192 | 13.8 | 2.11 | 2.25 | – |
| static int8_224 | 10.8 | 2.86 | 3.02 | – |
| static fp32_128 | 16.3 | 1.87 | 2.02 | – |
| static fp32_160 | 16.9 | 3.04 | 3.20 | – |
| static fp32_224 | 10.4 | 5.69 | 5.88 | – |
| **v2 chosen: int8_160 → fp32_128 (confidence<0.484)** | **16.5** | **2.69** | 3.54 | 63.1 |
| v2 live policy (online budget) | 16.5 | 2.85 | 3.54 | 71.6 |
| v2 accuracy-matched: int8_96 → int8_128 (confidence<0.121) | 8.0 | 0.85 | 1.60 | 26.8 |
| per-image oracle (upper bound) | 30.3 | 0.84 | 2.07 | – |

- vs Baseline (fp32_224): Δacc +6.2 pp [95% CI +4.4, +8.1], Δlatency -2.99 ms [-3.03, -2.95]
- vs AlwaysINT8 (int8_224): Δacc +5.8 pp [95% CI +4.1, +7.6], Δlatency -0.17 ms [-0.21, -0.12]
- v2 is faster than every point of the original INT8↔FP32 design space

## Condition: stressed

Latency source: ['profile']; budget = 4.17 ms (AlwaysINT8 mean latency)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static int8_96 | 6.4 | 0.86 | 3.65 | – |
| static int8_128 | 12.7 | 1.44 | 4.02 | – |
| static int8_160 | 14.9 | 2.26 | 5.32 | – |
| static int8_192 | 13.8 | 2.93 | 5.29 | – |
| static int8_224 | 10.8 | 4.09 | 6.07 | – |
| static fp32_128 | 16.3 | 2.75 | 5.29 | – |
| static fp32_160 | 16.9 | 4.30 | 6.43 | – |
| static fp32_224 | 10.4 | 7.78 | 13.92 | – |
| **v2 chosen: int8_160 → fp32_128 (confidence<0.484)** | **16.5** | **4.02** | 7.75 | 63.1 |
| v2 live policy (online budget) | 16.5 | 4.15 | 7.75 | 67.9 |
| v2 accuracy-matched: int8_96 → int8_128 (confidence<0.121) | 8.0 | 1.23 | 3.96 | 26.8 |
| per-image oracle (upper bound) | 30.3 | 1.23 | 3.99 | – |

- vs Baseline (fp32_224): Δacc +6.2 pp [95% CI +4.4, +8.1], Δlatency -3.76 ms [-3.94, -3.57]
- vs AlwaysINT8 (int8_224): Δacc +5.8 pp [95% CI +4.1, +7.6], Δlatency -0.08 ms [-0.21, +0.06]
- v2 is faster than every point of the original INT8↔FP32 design space

## Mixed schedule (40% idle → 60% stressed, like the original protocol)

| config | acc % | mean ms | p95 ms |
|---|---|---|---|
| Baseline fp32_224 | 10.4 | 7.00 | 12.32 |
| AlwaysINT8 int8_224 | 10.8 | 3.61 | 6.05 |
| **v2 resource-aware** | **16.5** | **3.63** | 7.00 |
