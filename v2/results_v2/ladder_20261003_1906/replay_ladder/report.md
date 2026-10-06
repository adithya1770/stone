# STONE v2 replay report (ladder, 150 ms budget)

Table: `v2/data/ladder/table_imagewoof.csv` — 3929 images (1979 tune / 1950 test, split by hash).

## How well does the cheap model's score flag its own mistakes? (AUROC, test split)

0.5 = no better than chance, 1.0 = perfect. High AUROC is what makes escalation pay off.

| variant | accuracy % | AUROC margin | AUROC confidence | AUROC -entropy |
|---|---|---|---|---|
| int8_224 | 57.2 | 0.782 | 0.788 | 0.790 |
| fp32_224 | 60.1 | 0.787 | 0.789 | 0.748 |
| mnv2q_160 | 55.2 | 0.737 | 0.758 | 0.749 |
| mnv2q_224 | 74.7 | 0.828 | 0.820 | 0.799 |
| efl0q_224 | 75.2 | 0.848 | 0.842 | 0.820 |
| efl1q_240 | 80.1 | 0.828 | 0.808 | 0.782 |
| efl2q_260 | 80.4 | 0.842 | 0.826 | 0.800 |
| efl3q_280 | 84.6 | 0.868 | 0.870 | 0.858 |
| efl4q_300 | 86.8 | 0.876 | 0.876 | 0.868 |

## Condition: idle

Latency source: ['profile']; budget = 150.00 ms (user; fastest config within 0.3 pp of the best)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static int8_224 | 57.2 | 50.57 | 93.86 | – |
| static fp32_224 | 60.1 | 69.85 | 113.83 | – |
| static mnv2q_160 | 55.2 | 24.96 | 23.83 | – |
| static mnv2q_224 | 74.7 | 50.73 | 114.95 | – |
| static efl0q_224 | 75.2 | 61.82 | 123.65 | – |
| static efl1q_240 | 80.1 | 90.33 | 150.96 | – |
| static efl2q_260 | 80.4 | 130.28 | 184.79 | – |
| static efl3q_280 | 84.6 | 187.02 | 247.69 | – |
| static efl4q_300 | 86.8 | 320.94 | 368.72 | – |
| **v2 chosen: mnv2q_224 → efl3q_280 (margin<0.440)** | **83.4** | **129.77** | 288.73 | 42.5 |
| v2 live policy (online budget) | 84.0 | 149.76 | 293.25 | 53.2 |
| v2 accuracy-matched: mnv2q_160 → mnv2q_224 (confidence<0.402) | 62.8 | 35.55 | 69.35 | 20.3 |
| per-image oracle (upper bound) | 93.2 | 44.17 | 116.96 | – |

- vs Baseline (fp32_224): Δacc +23.3 pp [95% CI +21.3, +25.3], Δlatency +59.92 ms [+55.80, +64.36]
- vs AlwaysINT8 (int8_224): Δacc +26.2 pp [95% CI +24.1, +28.2], Δlatency +79.20 ms [+75.14, +83.63]
- coin flip INT8↔FP32 at the same mean latency: 60.1% → v2 is +23.3 pp

## Condition: stressed

Latency source: ['profile']; budget = 150.00 ms (user; fastest config within 0.3 pp of the best)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static int8_224 | 57.2 | 91.40 | 95.52 | – |
| static fp32_224 | 60.1 | 132.05 | 135.06 | – |
| static mnv2q_160 | 55.2 | 47.87 | 52.19 | – |
| static mnv2q_224 | 74.7 | 92.25 | 100.08 | – |
| static efl0q_224 | 75.2 | 111.80 | 116.85 | – |
| static efl1q_240 | 80.1 | 163.17 | 173.83 | – |
| static efl2q_260 | 80.4 | 211.24 | 241.58 | – |
| static efl3q_280 | 84.6 | 273.42 | 361.17 | – |
| static efl4q_300 | 86.8 | 435.81 | 494.84 | – |
| **v2 chosen: mnv2q_224 → efl3q_280 (margin<0.144)** | **80.0** | **143.69** | 370.58 | 18.7 |
| v2 live policy (online budget) | 80.2 | 149.63 | 372.39 | 20.8 |
| v2 accuracy-matched: mnv2q_160 → mnv2q_224 (confidence<0.402) | 62.8 | 66.57 | 141.85 | 20.3 |
| per-image oracle (upper bound) | 93.2 | 77.99 | 170.46 | – |

- vs Baseline (fp32_224): Δacc +19.9 pp [95% CI +18.0, +21.9], Δlatency +11.64 ms [+6.77, +16.46]
- vs AlwaysINT8 (int8_224): Δacc +22.8 pp [95% CI +20.7, +24.8], Δlatency +52.29 ms [+47.44, +57.13]
- coin flip INT8↔FP32 at the same mean latency: 60.1% → v2 is +19.9 pp

## Mixed schedule (40% idle → 60% stressed, like the original protocol)

| config | acc % | mean ms | p95 ms |
|---|---|---|---|
| Baseline fp32_224 | 60.1 | 107.53 | 134.77 |
| AlwaysINT8 int8_224 | 57.2 | 74.80 | 95.52 |
| **v2 resource-aware** | **81.6** | **148.26** | 364.14 |
