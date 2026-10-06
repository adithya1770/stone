# STONE v2 replay report (tiny, rebuilt)

Table: `v2/results_v2/validate_rebuilt_arm/table_tiny.csv` — 1200 images (597 tune / 603 test, split by hash).

## How well does the cheap model's score flag its own mistakes? (AUROC, test split)

0.5 = no better than chance, 1.0 = perfect. High AUROC is what makes escalation pay off.

| variant | accuracy % | AUROC margin | AUROC confidence | AUROC -entropy |
|---|---|---|---|---|
| int8_224 | 9.5 | 0.709 | 0.728 | 0.744 |
| fp32_224 | 9.0 | 0.690 | 0.687 | 0.652 |
| mnv2f_128 | 23.1 | 0.761 | 0.775 | 0.788 |
| mnv2f_160 | 28.4 | 0.694 | 0.701 | 0.701 |
| mnv2f_224 | 19.1 | 0.675 | 0.700 | 0.696 |
| mnv2q_128 | 20.2 | 0.612 | 0.667 | 0.699 |
| mnv2q_160 | 27.5 | 0.676 | 0.712 | 0.723 |
| mnv2q_224 | 19.7 | 0.696 | 0.704 | 0.708 |

## Condition: idle

Latency source: ['profile']; budget = 2.83 ms (AlwaysINT8 mean latency)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static int8_224 | 9.5 | 2.83 | 3.04 | – |
| static fp32_224 | 9.0 | 5.59 | 5.73 | – |
| static mnv2f_128 | 23.1 | 2.98 | 3.09 | – |
| static mnv2f_160 | 28.4 | 4.32 | 4.61 | – |
| static mnv2f_224 | 19.1 | 7.95 | 8.22 | – |
| static mnv2q_128 | 20.2 | 0.96 | 1.08 | – |
| static mnv2q_160 | 27.5 | 1.48 | 1.57 | – |
| static mnv2q_224 | 19.7 | 2.84 | 3.09 | – |
| **v2 chosen: mnv2q_160 → mnv2f_160 (margin<0.066)** | **28.4** | **2.26** | 5.81 | 18.2 |
| v2 live policy (online budget) | 28.7 | 2.81 | 5.87 | 31.0 |
| v2 accuracy-matched: mnv2q_128 | 20.2 | 0.96 | 1.08 | 0.0 |
| per-image oracle (upper bound) | 39.6 | 1.21 | 2.89 | – |

- vs Baseline (fp32_224): Δacc +19.4 pp [95% CI +15.9, +22.9], Δlatency -3.32 ms [-3.45, -3.19]
- vs AlwaysINT8 (int8_224): Δacc +18.9 pp [95% CI +15.6, +22.4], Δlatency -0.57 ms [-0.70, -0.43]
- v2 is faster than every point of the original INT8↔FP32 design space

## Condition: stressed

Latency source: ['profile']; budget = 4.38 ms (AlwaysINT8 mean latency)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static int8_224 | 9.5 | 4.24 | 9.37 | – |
| static fp32_224 | 9.0 | 7.86 | 12.44 | – |
| static mnv2f_128 | 23.1 | 4.98 | 9.21 | – |
| static mnv2f_160 | 28.4 | 6.33 | 10.62 | – |
| static mnv2f_224 | 19.1 | 11.96 | 19.93 | – |
| static mnv2q_128 | 20.2 | 1.31 | 4.01 | – |
| static mnv2q_160 | 27.5 | 2.24 | 4.59 | – |
| static mnv2q_224 | 19.7 | 4.00 | 6.09 | – |
| **v2 chosen: mnv2q_160 → mnv2f_160 (margin<0.155)** | **28.5** | **4.35** | 12.21 | 32.2 |
| v2 live policy (online budget) | 29.0 | 4.56 | 12.21 | 35.7 |
| v2 accuracy-matched: mnv2q_128 | 20.2 | 1.31 | 4.01 | 0.0 |
| per-image oracle (upper bound) | 39.6 | 1.70 | 4.57 | – |

- vs Baseline (fp32_224): Δacc +19.6 pp [95% CI +16.3, +22.9], Δlatency -3.51 ms [-3.85, -3.15]
- vs AlwaysINT8 (int8_224): Δacc +19.1 pp [95% CI +15.8, +22.6], Δlatency +0.11 ms [-0.21, +0.43]
- coin flip INT8↔FP32 at the same mean latency: 9.4% → v2 is +19.1 pp

## Mixed schedule (40% idle → 60% stressed, like the original protocol)

| config | acc % | mean ms | p95 ms |
|---|---|---|---|
| Baseline fp32_224 | 9.0 | 6.96 | 12.39 |
| AlwaysINT8 int8_224 | 9.5 | 3.63 | 9.37 |
| **v2 resource-aware** | **29.0** | **3.85** | 9.63 |
