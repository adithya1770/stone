# STONE v2 replay report (incl. fixed preprocessing)

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
| fp32n_128 | 57.8 | 0.845 | 0.852 | 0.860 |
| fp32n_160 | 72.0 | 0.870 | 0.874 | 0.866 |
| fp32n_224 | 81.5 | 0.890 | 0.884 | 0.835 |

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
| static fp32n_128 | 57.8 | 28.08 | 57.41 | – |
| static fp32n_160 | 72.0 | 41.02 | 88.96 | – |
| static fp32n_224 | 81.5 | 79.48 | 129.19 | – |
| **v2 chosen: fp32n_160 → fp32n_224 (confidence<0.521)** | **77.6** | **56.03** | 150.05 | 18.5 |
| v2 live policy (online budget) | 78.3 | 57.74 | 151.63 | 20.5 |
| v2 accuracy-matched: fp32n_128 → fp32n_224 (margin<0.264) | 66.1 | 41.75 | 122.64 | 16.8 |
| per-image oracle (upper bound) | 87.6 | 26.86 | 66.64 | – |

- vs Baseline (fp32_224): Δacc +8.3 pp [95% CI +4.9, +12.0], Δlatency -19.17 ms [-23.11, -14.91]
- vs AlwaysINT8 (int8_224): Δacc +11.0 pp [95% CI +7.1, +15.1], Δlatency +1.18 ms [-2.99, +5.47]
- coin flip INT8↔FP32 at the same mean latency: 66.7% → v2 is +10.8 pp

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
| static fp32n_128 | 57.8 | 45.53 | 47.99 | – |
| static fp32n_160 | 72.0 | 71.48 | 81.40 | – |
| static fp32n_224 | 81.5 | 141.03 | 183.07 | – |
| **v2 chosen: fp32n_160 → fp32n_224 (confidence<0.494)** | **77.3** | **94.85** | 213.36 | 16.6 |
| v2 live policy (online budget) | 77.3 | 97.77 | 214.44 | 18.5 |
| v2 accuracy-matched: fp32n_128 → fp32n_224 (margin<0.264) | 66.1 | 69.39 | 186.57 | 16.8 |
| per-image oracle (upper bound) | 87.6 | 45.49 | 129.09 | – |

- vs Baseline (fp32_224): Δacc +8.0 pp [95% CI +4.4, +11.7], Δlatency -42.98 ms [-47.94, -37.90]
- vs AlwaysINT8 (int8_224): Δacc +10.7 pp [95% CI +6.8, +14.9], Δlatency -2.04 ms [-7.20, +3.08]
- v2 is faster than every point of the original INT8↔FP32 design space

## Mixed schedule (40% idle → 60% stressed, like the original protocol)

| config | acc % | mean ms | p95 ms |
|---|---|---|---|
| Baseline fp32_224 | 69.3 | 113.07 | 143.92 |
| AlwaysINT8 int8_224 | 66.6 | 80.56 | 106.47 |
| **v2 resource-aware** | **78.0** | **81.88** | 209.47 |
