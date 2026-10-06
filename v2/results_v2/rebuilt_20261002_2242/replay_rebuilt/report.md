# STONE v2 replay report (rebuilt INT8 + FP32)

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

Latency source: ['profile']; budget = 56.05 ms (AlwaysINT8 mean latency)

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
| **v2 chosen: mnv2q_160 → mnv2f_224 (entropy<-0.993)** | **80.2** | **57.39** | 140.37 | 34.6 |
| v2 live policy (online budget) | 80.2 | 56.00 | 140.11 | 32.7 |
| v2 accuracy-matched: mnv2q_160 → mnv2q_224 (confidence<0.228) | 70.0 | 33.32 | 68.73 | 3.9 |
| per-image oracle (upper bound) | 87.1 | 27.77 | 63.12 | – |

- vs Baseline (fp32_224): Δacc +11.0 pp [95% CI +7.3, +14.6], Δlatency -13.75 ms [-18.23, -9.41]
- vs AlwaysINT8 (int8_224): Δacc +13.7 pp [95% CI +10.0, +17.6], Δlatency +3.25 ms [-1.16, +7.78]
- coin flip INT8↔FP32 at the same mean latency: 67.1% → v2 is +13.1 pp

## Condition: stressed

Latency source: ['profile']; budget = 93.13 ms (AlwaysINT8 mean latency)

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
| **v2 chosen: mnv2q_224** | **81.7** | **91.40** | 101.75 | 0.0 |
| v2 live policy (online budget) | 81.7 | 91.40 | 101.75 | 0.0 |
| v2 accuracy-matched: mnv2q_160 → mnv2q_224 (entropy<-2.844) | 69.3 | 50.75 | 54.28 | 1.7 |
| per-image oracle (upper bound) | 87.1 | 44.51 | 95.04 | – |

- vs Baseline (fp32_224): Δacc +12.4 pp [95% CI +9.0, +16.1], Δlatency -45.68 ms [-47.45, -43.99]
- vs AlwaysINT8 (int8_224): Δacc +15.1 pp [95% CI +11.7, +19.0], Δlatency -2.21 ms [-3.23, -1.23]
- v2 is faster than every point of the original INT8↔FP32 design space

## Mixed schedule (40% idle → 60% stressed, like the original protocol)

| config | acc % | mean ms | p95 ms |
|---|---|---|---|
| Baseline fp32_224 | 69.3 | 111.35 | 142.29 |
| AlwaysINT8 int8_224 | 66.6 | 78.41 | 109.61 |
| **v2 resource-aware** | **81.5** | **76.76** | 101.75 |
