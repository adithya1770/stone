# STONE v2 replay report (all models)

Table: `v2/results_v2/rebuilt_20261002_2242/table_imagenette.csv` — 800 images (390 tune / 410 test, split by hash).

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
| fp32n_128 | 57.8 | 0.845 | 0.852 | 0.860 |
| fp32n_160 | 72.0 | 0.870 | 0.874 | 0.866 |
| fp32n_224 | 81.5 | 0.890 | 0.884 | 0.835 |
| mnv2f_128 | 59.0 | 0.844 | 0.850 | 0.857 |
| mnv2f_160 | 73.2 | 0.867 | 0.877 | 0.874 |
| mnv2f_224 | 83.4 | 0.893 | 0.883 | 0.832 |
| mnv2q_128 | 43.2 | 0.618 | 0.708 | 0.747 |
| mnv2q_160 | 69.0 | 0.835 | 0.858 | 0.860 |
| mnv2q_224 | 81.7 | 0.876 | 0.870 | 0.865 |

## Condition: idle

Latency source: ['profile']; budget = 55.08 ms (AlwaysINT8 mean latency)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static int8_96 | 12.0 | 11.24 | 25.74 | – |
| static int8_128 | 29.8 | 19.53 | 43.95 | – |
| static int8_160 | 54.4 | 27.54 | 59.38 | – |
| static int8_192 | 63.9 | 38.77 | 85.34 | – |
| static int8_224 | 66.6 | 53.93 | 115.02 | – |
| static fp32_128 | 42.2 | 27.97 | 57.09 | – |
| static fp32_160 | 59.3 | 43.66 | 89.06 | – |
| static fp32_224 | 69.3 | 70.70 | 117.31 | – |
| static fp32n_128 | 57.8 | 26.58 | 56.91 | – |
| static fp32n_160 | 72.0 | 47.25 | 88.89 | – |
| static fp32n_224 | 81.5 | 75.56 | 128.99 | – |
| static mnv2f_128 | 59.0 | 29.41 | 62.61 | – |
| static mnv2f_160 | 73.2 | 46.28 | 99.09 | – |
| static mnv2f_224 | 83.4 | 77.36 | 135.72 | – |
| static mnv2q_128 | 43.2 | 21.18 | 44.06 | – |
| static mnv2q_160 | 69.0 | 30.80 | 68.70 | – |
| static mnv2q_224 | 81.7 | 61.31 | 117.59 | – |
| **v2 chosen: mnv2q_160 → mnv2f_224 (confidence<0.568)** | **79.5** | **55.94** | 131.29 | 32.9 |
| v2 live policy (online budget) | 79.0 | 54.38 | 130.76 | 31.0 |
| v2 accuracy-matched: mnv2q_160 → mnv2q_224 (entropy<-2.844) | 69.3 | 31.90 | 68.71 | 1.7 |
| per-image oracle (upper bound) | 88.3 | 23.08 | 56.71 | – |

- vs Baseline (fp32_224): Δacc +10.2 pp [95% CI +6.8, +13.7], Δlatency -14.76 ms [-18.91, -10.59]
- vs AlwaysINT8 (int8_224): Δacc +12.9 pp [95% CI +9.3, +16.6], Δlatency +2.00 ms [-2.21, +6.61]
- coin flip INT8↔FP32 at the same mean latency: 66.9% → v2 is +12.6 pp

## Condition: stressed

Latency source: ['profile']; budget = 93.61 ms (AlwaysINT8 mean latency)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static int8_96 | 12.0 | 17.88 | 21.74 | – |
| static int8_128 | 29.8 | 31.44 | 36.70 | – |
| static int8_160 | 54.4 | 49.78 | 55.08 | – |
| static int8_192 | 63.9 | 69.81 | 74.39 | – |
| static int8_224 | 66.6 | 93.03 | 100.17 | – |
| static fp32_128 | 42.2 | 43.38 | 46.92 | – |
| static fp32_160 | 59.3 | 69.09 | 73.44 | – |
| static fp32_224 | 69.3 | 135.87 | 142.29 | – |
| static fp32n_128 | 57.8 | 44.12 | 47.60 | – |
| static fp32n_160 | 72.0 | 68.71 | 73.61 | – |
| static fp32n_224 | 81.5 | 133.95 | 142.85 | – |
| static mnv2f_128 | 59.0 | 51.81 | 55.48 | – |
| static mnv2f_160 | 73.2 | 79.61 | 85.67 | – |
| static mnv2f_224 | 83.4 | 150.63 | 161.45 | – |
| static mnv2q_128 | 43.2 | 31.62 | 36.01 | – |
| static mnv2q_160 | 69.0 | 49.11 | 54.20 | – |
| static mnv2q_224 | 81.7 | 92.47 | 101.75 | – |
| **v2 chosen: mnv2q_224 → int8_224 (confidence<0.123)** | **81.7** | **94.32** | 102.62 | 2.0 |
| v2 live policy (online budget) | 81.7 | 94.33 | 102.62 | 2.0 |
| v2 accuracy-matched: mnv2q_160 → mnv2q_224 (confidence<0.228) | 70.0 | 52.70 | 54.34 | 3.9 |
| per-image oracle (upper bound) | 88.3 | 37.95 | 93.25 | – |

- vs Baseline (fp32_224): Δacc +12.4 pp [95% CI +9.0, +16.1], Δlatency -41.56 ms [-43.37, -39.70]
- vs AlwaysINT8 (int8_224): Δacc +15.1 pp [95% CI +11.7, +19.0], Δlatency +1.29 ms [-0.16, +2.87]
- coin flip INT8↔FP32 at the same mean latency: 66.7% → v2 is +15.0 pp

## Mixed schedule (40% idle → 60% stressed, like the original protocol)

| config | acc % | mean ms | p95 ms |
|---|---|---|---|
| Baseline fp32_224 | 69.3 | 110.02 | 141.61 |
| AlwaysINT8 int8_224 | 66.6 | 78.00 | 105.30 |
| **v2 resource-aware** | **81.2** | **78.30** | 102.62 |
