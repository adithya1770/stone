# STONE v2 replay report (original preprocessing)

Table: `v2/results_v2/pi_20261002_1807/table_imagenette.csv` — 800 images (390 tune / 410 test, split by hash).

## How well does the cheap model's score flag its own mistakes? (AUROC, test split)

0.5 = no better than chance, 1.0 = perfect. High AUROC is what makes escalation pay off.

| variant | accuracy % | AUROC margin | AUROC confidence | AUROC -entropy |
|---|---|---|---|---|
| int8_96 | 12.0 | 0.548 | 0.683 | 0.741 |
| int8_128 | 29.8 | 0.514 | 0.632 | 0.699 |
| int8_160 | 54.4 | 0.687 | 0.772 | 0.799 |
| int8_192 | 63.9 | 0.809 | 0.824 | 0.818 |
| int8_224 | 66.6 | 0.871 | 0.868 | 0.852 |
| fp32_128 | 42.2 | 0.845 | 0.852 | 0.854 |
| fp32_160 | 59.3 | 0.868 | 0.873 | 0.860 |
| fp32_224 | 69.3 | 0.861 | 0.859 | 0.807 |

## Condition: idle

Latency source: ['profile']; budget = 57.58 ms (AlwaysINT8 mean latency)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static int8_96 | 12.0 | 12.57 | 25.87 | – |
| static int8_128 | 29.8 | 19.75 | 44.07 | – |
| static int8_160 | 54.4 | 34.39 | 68.52 | – |
| static int8_192 | 63.9 | 39.98 | 86.74 | – |
| static int8_224 | 66.6 | 54.85 | 114.28 | – |
| static fp32_128 | 42.2 | 24.02 | 49.53 | – |
| static fp32_160 | 59.3 | 43.33 | 88.97 | – |
| static fp32_224 | 69.3 | 75.20 | 126.72 | – |
| **v2 chosen: int8_224** | **66.6** | **54.85** | 114.28 | 0.0 |
| v2 live policy (online budget) | 66.6 | 54.85 | 114.28 | 0.0 |
| v2 accuracy-matched: int8_224 | 66.6 | 54.85 | 114.28 | 0.0 |
| per-image oracle (upper bound) | 78.5 | 24.45 | 68.02 | – |

- vs Baseline (fp32_224): Δacc -2.7 pp [95% CI -5.1, -0.5], Δlatency -20.36 ms [-23.40, -17.21]
- vs AlwaysINT8 (int8_224): Δacc +0.0 pp [95% CI +0.0, +0.0], Δlatency +0.00 ms [+0.00, +0.00]
- coin flip INT8↔FP32 at the same mean latency: 66.6% → v2 is +0.0 pp

## Condition: stressed

Latency source: ['profile']; budget = 97.87 ms (AlwaysINT8 mean latency)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static int8_96 | 12.0 | 20.77 | 22.46 | – |
| static int8_128 | 29.8 | 34.07 | 37.60 | – |
| static int8_160 | 54.4 | 50.26 | 55.39 | – |
| static int8_192 | 63.9 | 71.51 | 83.35 | – |
| static int8_224 | 66.6 | 96.90 | 106.47 | – |
| static fp32_128 | 42.2 | 44.04 | 47.70 | – |
| static fp32_160 | 59.3 | 72.08 | 77.35 | – |
| static fp32_224 | 69.3 | 137.84 | 149.61 | – |
| **v2 chosen: int8_224** | **66.6** | **96.90** | 106.47 | 0.0 |
| v2 live policy (online budget) | 66.6 | 96.90 | 106.47 | 0.0 |
| v2 accuracy-matched: int8_224 | 66.6 | 96.90 | 106.47 | 0.0 |
| per-image oracle (upper bound) | 78.5 | 41.82 | 89.56 | – |

- vs Baseline (fp32_224): Δacc -2.7 pp [95% CI -5.1, -0.5], Δlatency -40.94 ms [-42.01, -39.77]
- vs AlwaysINT8 (int8_224): Δacc +0.0 pp [95% CI +0.0, +0.0], Δlatency +0.00 ms [+0.00, +0.00]
- coin flip INT8↔FP32 at the same mean latency: 66.6% → v2 is +0.0 pp

## Mixed schedule (40% idle → 60% stressed, like the original protocol)

| config | acc % | mean ms | p95 ms |
|---|---|---|---|
| Baseline fp32_224 | 69.3 | 113.07 | 143.92 |
| AlwaysINT8 int8_224 | 66.6 | 80.56 | 106.47 |
| **v2 resource-aware** | **66.6** | **80.56** | 106.47 |
