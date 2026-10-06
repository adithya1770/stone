# STONE v2 replay report (v2_final: rebuilt models, per-state budgets)

Table: `v2/results_v2/rebuilt_20261002_2242/table_imagenette.csv` — 800 images (390 tune / 410 test, split by hash).

## How well does the cheap model's score flag its own mistakes? (AUROC, test split)

0.5 = no better than chance, 1.0 = perfect. High AUROC is what makes escalation pay off.

| variant | accuracy % | AUROC margin | AUROC confidence | AUROC -entropy |
|---|---|---|---|---|
| int8_224 | 66.6 | 0.871 | 0.868 | 0.852 |
| fp32_224 | 69.3 | 0.861 | 0.859 | 0.807 |
| mnv2f_128 | 59.0 | 0.844 | 0.850 | 0.857 |
| mnv2f_160 | 73.2 | 0.867 | 0.877 | 0.874 |
| mnv2f_224 | 83.4 | 0.893 | 0.883 | 0.832 |
| mnv2q_128 | 43.2 | 0.618 | 0.708 | 0.747 |
| mnv2q_160 | 69.0 | 0.835 | 0.858 | 0.860 |
| mnv2q_224 | 81.7 | 0.876 | 0.870 | 0.865 |

## Condition: idle

Latency source: ['profile']; budget = 81.30 ms (mean latency of mnv2f_224 + 5% tolerance; fastest config within 0.5 pp of the best)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static int8_224 | 66.6 | 54.14 | 115.02 | – |
| static fp32_224 | 69.3 | 71.14 | 117.64 | – |
| static mnv2f_128 | 59.0 | 30.93 | 63.10 | – |
| static mnv2f_160 | 73.2 | 44.50 | 99.09 | – |
| static mnv2f_224 | 83.4 | 78.27 | 135.72 | – |
| static mnv2q_128 | 43.2 | 19.74 | 43.98 | – |
| static mnv2q_160 | 69.0 | 31.03 | 68.70 | – |
| static mnv2q_224 | 81.7 | 58.54 | 117.59 | – |
| **v2 chosen: mnv2q_160 → mnv2f_224 (margin<0.773)** | **82.4** | **74.85** | 141.01 | 57.1 |
| v2 live policy (online budget) | 82.7 | 79.81 | 155.13 | 63.2 |
| v2 accuracy-matched: mnv2q_160 → mnv2q_224 (confidence<0.228) | 70.0 | 33.32 | 68.73 | 3.9 |
| per-image oracle (upper bound) | 87.1 | 27.77 | 63.12 | – |

- vs Baseline (fp32_224): Δacc +13.2 pp [95% CI +9.8, +16.8], Δlatency +3.70 ms [-0.83, +8.18]
- vs AlwaysINT8 (int8_224): Δacc +15.9 pp [95% CI +12.2, +19.5], Δlatency +20.70 ms [+15.85, +25.19]
- coin flip INT8↔FP32 at the same mean latency: 69.3% → v2 is +13.2 pp

## Condition: stressed

Latency source: ['profile']; budget = 96.43 ms (mean latency of mnv2q_224 + 5% tolerance; fastest config within 0.5 pp of the best)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static int8_224 | 66.6 | 93.61 | 100.17 | – |
| static fp32_224 | 69.3 | 137.08 | 198.89 | – |
| static mnv2f_128 | 59.0 | 52.08 | 55.48 | – |
| static mnv2f_160 | 73.2 | 79.56 | 85.67 | – |
| static mnv2f_224 | 83.4 | 149.82 | 161.15 | – |
| static mnv2q_128 | 43.2 | 31.38 | 35.43 | – |
| static mnv2q_160 | 69.0 | 49.16 | 54.20 | – |
| static mnv2q_224 | 81.7 | 91.40 | 101.75 | – |
| **v2 chosen: mnv2q_224 → mnv2f_224 (entropy<-3.325)** | **82.0** | **94.64** | 102.62 | 2.2 |
| v2 live policy (online budget) | 82.0 | 95.36 | 103.34 | 2.7 |
| v2 accuracy-matched: mnv2q_160 → mnv2q_224 (entropy<-2.844) | 69.3 | 50.75 | 54.28 | 1.7 |
| per-image oracle (upper bound) | 87.1 | 44.51 | 95.04 | – |

- vs Baseline (fp32_224): Δacc +12.7 pp [95% CI +9.3, +16.3], Δlatency -42.44 ms [-45.07, -39.68]
- vs AlwaysINT8 (int8_224): Δacc +15.4 pp [95% CI +12.0, +19.3], Δlatency +1.03 ms [-1.08, +3.44]
- coin flip INT8↔FP32 at the same mean latency: 66.6% → v2 is +15.3 pp

## Mixed schedule (40% idle → 60% stressed, like the original protocol)

| config | acc % | mean ms | p95 ms |
|---|---|---|---|
| Baseline fp32_224 | 69.3 | 111.35 | 142.29 |
| AlwaysINT8 int8_224 | 66.6 | 78.41 | 109.61 |
| **v2 resource-aware** | **82.7** | **87.14** | 127.53 |
