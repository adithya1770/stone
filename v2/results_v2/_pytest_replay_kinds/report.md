# STONE v2 replay report 

Table: `/sessions/rcw-017l9fs4cfqvzatvlekujv5d/tmp/pytest-of-rcw-017l9fs4cfqvzatvlekujv5d/pytest-28/test_replay_main_with_kinds_ke0/t2.csv` — 400 images (203 tune / 197 test, split by hash).

## How well does the cheap model's score flag its own mistakes? (AUROC, test split)

0.5 = no better than chance, 1.0 = perfect. High AUROC is what makes escalation pay off.

| variant | accuracy % | AUROC margin | AUROC confidence | AUROC -entropy |
|---|---|---|---|---|
| mnv2q_128 | 71.6 | 1.000 | 0.500 | 0.500 |
| int8_224 | 85.3 | 0.500 | 0.500 | 0.500 |
| fp32_224 | 97.5 | 0.500 | 0.500 | 0.500 |

## Condition: idle

Latency source: ['table']; budget = 30.00 ms (mean latency of int8_224)

| config | acc % (test) | mean ms | p95 ms | escalated % |
|---|---|---|---|---|
| static mnv2q_128 | 71.6 | 10.00 | 10.00 | – |
| static int8_224 | 85.3 | 30.00 | 30.00 | – |
| static fp32_224 | 97.5 | 50.00 | 50.00 | – |
| **v2 chosen: mnv2q_128** | **71.6** | **10.00** | 10.00 | 0.0 |
| v2 live policy (online budget) | 71.6 | 10.00 | 10.00 | 0.0 |
| per-image oracle (upper bound) | 71.6 | 10.00 | 10.00 | – |

- vs Baseline (fp32_224): Δacc -25.9 pp [95% CI -33.0, -19.3], Δlatency -40.00 ms [-40.00, -40.00]
- vs AlwaysINT8 (int8_224): Δacc -13.7 pp [95% CI -18.8, -9.1], Δlatency -20.00 ms [-20.00, -20.00]
- v2 is faster than every point of the original INT8↔FP32 design space
