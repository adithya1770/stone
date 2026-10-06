# STONE v2 replay report (tiny, fixed preprocessing)

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
| fp32n_128 | 25.8 | 0.721 | 0.745 | 0.767 |
| fp32n_160 | 27.0 | 0.715 | 0.732 | 0.737 |
| fp32n_224 | 18.3 | 0.716 | 0.732 | 0.720 |

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
| static fp32n_128 | 25.8 | 1.87 | 2.00 | – |
| static fp32n_160 | 27.0 | 3.03 | 3.18 | – |
| static fp32n_224 | 18.3 | 5.68 | 5.84 | – |
| **v2 chosen: fp32n_128 → fp32n_160 (margin<0.136)** | **26.9** | **2.54** | 4.98 | 22.2 |
| v2 live policy (online budget) | 27.0 | 2.85 | 5.00 | 32.6 |
| v2 accuracy-matched: int8_96 → fp32n_128 (confidence<0.078) | 8.4 | 0.86 | 2.45 | 14.1 |
| per-image oracle (upper bound) | 41.8 | 1.05 | 2.95 | – |

- vs Baseline (fp32_224): Δacc +16.5 pp [95% CI +14.3, +18.7], Δlatency -3.15 ms [-3.21, -3.09]
- vs AlwaysINT8 (int8_224): Δacc +16.1 pp [95% CI +14.0, +18.3], Δlatency -0.32 ms [-0.39, -0.26]
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
| static fp32n_128 | 25.8 | 2.69 | 5.48 | – |
| static fp32n_160 | 27.0 | 4.58 | 9.21 | – |
| static fp32n_224 | 18.3 | 8.27 | 12.50 | – |
| **v2 chosen: fp32n_128 → fp32n_160 (margin<0.136)** | **26.9** | **3.71** | 8.32 | 22.2 |
| v2 live policy (online budget) | 27.1 | 4.15 | 8.73 | 32.1 |
| v2 accuracy-matched: int8_96 → fp32n_128 (confidence<0.078) | 8.4 | 1.25 | 3.96 | 14.1 |
| per-image oracle (upper bound) | 41.8 | 1.54 | 5.11 | – |

- vs Baseline (fp32_224): Δacc +16.5 pp [95% CI +14.3, +18.7], Δlatency -4.07 ms [-4.25, -3.88]
- vs AlwaysINT8 (int8_224): Δacc +16.1 pp [95% CI +14.0, +18.3], Δlatency -0.38 ms [-0.53, -0.24]
- v2 is faster than every point of the original INT8↔FP32 design space

## Mixed schedule (40% idle → 60% stressed, like the original protocol)

| config | acc % | mean ms | p95 ms |
|---|---|---|---|
| Baseline fp32_224 | 10.4 | 7.00 | 12.32 |
| AlwaysINT8 int8_224 | 10.8 | 3.61 | 6.05 |
| **v2 resource-aware** | **27.2** | **3.60** | 8.23 |
