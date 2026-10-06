# STONE v2 — resource-aware *and* input-aware model selection

v2 extends the original STONE runtime (Raspberry Pi 4, MobileNetV2) without
changing a single original file. Everything lives in `v2/`; deleting this
folder returns the repo exactly to the original. v2 imports the original
`runtime/` code read-only and writes only inside `v2/`.

## Why v2: what the analysis of the original results showed

Recomputed from `results/stone_pi_multitrial2` (the data behind the paper's Table I):

| finding | evidence |
|---|---|
| The adaptive controllers behave like a weighted coin flip between INT8 and FP32 | A coin flip at EightSignal's INT8 rate (72.2%) predicts **102.9 ms / 70.2%**; the paper reports **102.9 ms / 70.0%**. LinUCB likewise sits on the coin-flip line. |
| They cannot make per-request decisions | LinUCB's context is `[cpu, ram, temp, battery, 1]` — nothing about the image. EightSignal's "difficulty" signal is the *previous* image's confidence. |
| The reward gives the bandit almost nothing to learn from | Reward gap FP32−INT8 = −0.0075 vs per-request std 0.215 (SNR 0.035): ~800 samples needed, runs have 100. The latency term `max(0, 1−ms/80)` is 0 for 64% of Pi requests (Pi latencies are ~90–130 ms). |
| Per-image headroom is large | On the 60 stressed images: INT8 is right whenever FP32 is right on 56; only **4 images need FP32**. A per-image oracle gets 75% accuracy at ~95 ms. |
| Accuracy differences in Table I are within noise | Same 60 images every trial, so "± 0.0" only shows runs agree; the real 95% CI is about **±11 pp**. Latency differences are real. |
| "FP32" is not FP32 | `mobilenet_v2_fp32.tflite` (3.8 MB) contains 64 int8 weight tensors (dynamic-range quantized); a float MobileNetV2 is ~14 MB. |
| Preprocessing bug (confirmed on the Pi) | The float model expects the Keras `img/127.5−1` input; the original feeds `img/255`. Same model file: **69.6% → 82.4%** on 500 Imagenette photos, **73.3% → 85.0%** on 300 unseen stressed photos (41 gained / 6 lost, p < 0.001). |
| The custom dataset carries no signal | 14 images cycled to 60 rows, 24 unlabelled; every config scores exactly 50.0%. |

## How v2 decides, per request

1. **Device check (resource-aware).** Telemetry is read in a background thread (the original
   `get_telemetry()` blocks 0.5 s per call). `ResourceMonitor` maps it to a tier:
   `idle`, `stressed` (CPU contention / heat / frequency drop) or `critical`
   (≥ 85 °C or the Pi firmware's "currently throttled" flag → the stressed tier's first model, never escalate).
   It can also run on the original EightSignal vote score (`--monitor eightsignal`).
2. **Cheap first look.** Each tier has its own ladder: a cheap first model and an
   escalation model, chosen from a resolution ladder of the same weights
   (`int8_96 … int8_224`, `fp32_128 … fp32_224`): MobileNetV2 is fully convolutional,
   so one `.tflite` runs at several input sizes. INT8 at 128 px costs ~1/3 of INT8 at 224 px.
3. **Image check (input-aware).** If the cheap model's uncertainty score (margin p1−p2,
   confidence p1, or −entropy — whichever flags errors best, chosen offline per tier)
   is below θ, the image is escalated.
4. **Latency budget.** θ is adapted online: `θ ← θ + η·(budget − latency)/budget`.
   Over budget → fewer escalations. No labels needed, so it works in the live server.
   Default budget is *relative*: "spend on average k× the cheap model's live cost",
   with k from offline tuning. It carries over from back-to-back profiling to the
   protocol's 0.5 s gaps and from idle to stressed. `--budget-mode absolute` gives a hard SLO.

Operating points (ladder, signal, θ, budget per tier) are tuned offline on a **held-out
half** of the images, evaluated on the other half, and written to
`v2/configs/operating_points.json`.

## Folder layout

```
v2/
  common/      engine.py (multi-variant TFLite engine, bit-identical to the original at 224 px),
               labels.py (index-based correctness), datasets.py, paths.py (refuses writes outside v2/)
  routing/     telemetry.py, resource_state.py, budget.py, policies.py, frame_gate.py
  bench/       profile_variants.py   per-variant latency (idle / stressed, raw samples)
               build_response_table.py  every variant × every image -> one CSV
               replay_eval.py        exact offline policy evaluation, Pareto plots, op-point selection
               check_preprocessing.py
  variants/    build_variants.py     optional: true FP32 + correctly calibrated INT8 + MobileNetV3-Small
  run_suite_v2.py   the original Pi protocol, original controllers + v2 side by side
  analyse_v2.py     Table-I style comparison with CIs, McNemar test, coin-flip line
  server_v2.py      live demo server (port 8001), same page/endpoints as web_demo/server.py
  tests/            30 unit tests (run anywhere, no Pi needed)
  pi_run_all.sh     everything on the Pi, one command
  mac_validate.sh   pipeline check without Pi/imagenette (tiny-imagenet)
```

## Run it on the Raspberry Pi (≈ 60–75 min)

```bash
cd stone
sudo apt install stress-ng          # same stressor as the original experiments
python3 -m pip install -r v2/requirements.txt
bash v2/pi_run_all.sh               # TRIALS=3 LIMIT=800 by default
```

Needs `datasets/imagenette2-320/val` (as for the original runs). Steps: unit tests →
preprocessing check → latency profile idle + stressed → response table on 800 held-out
imagenette images (the original 100-image sequence is excluded) → replay → 3 trials of
the original protocol → analysis. Read:

- `v2/results_v2/suite/<tag>/analysis/results.md` — **the Table-I comparison**
- `v2/results_v2/<tag>/replay_original/report.md` — offline evaluation, original preprocessing (like-for-like)
- `v2/results_v2/<tag>/replay_fixed/report.md` — including the preprocessing fix (separate ablation)

Live demo: `python3 -m v2.server_v2` → `http://<pi>:8001` (original demo stays on 8000).

## Fairness rules built in

- **Like-for-like by default:** the main v2 config uses only variants with the original
  preprocessing. Gains from the preprocessing fix (`fp32n_*`, `baseline_fixed`,
  `v2_resource_aware_fixed`) are always reported separately.
- **No tuning on test images:** thresholds come from a hash-split tune half; the original
  100-image protocol sequence is excluded from the tuning table.
- **Same protocol, same session:** Baseline, AlwaysINT8, LinUCB and EightSignal are re-run
  next to v2 (fresh state, cool-down between configs), with the original stress schedule
  and the original "stressed = cpu ≥ 70%" filter.
- **Honest uncertainty:** Wilson CIs on stressed images, paired McNemar test vs Baseline,
  bootstrap CIs in replay, and the coin-flip line as the minimum bar.

## Results on the Raspberry Pi 4 (2 Oct 2026)

Full write-up: the "STONE v2: findings and Raspberry Pi results" doc. Stressed rows only.

**Original protocol** (his 100-image sequence, 60 stressed photos, 3 trials, `suite/pi_run2`):
his configs reproduce Table I (Baseline 75.0%, AlwaysINT8 68.3%, latency within 1-3 ms).
v2 (fixed preprocessing) gets 75.0% at 86.0 ms vs Baseline 75.0% at 131.5 ms and
AlwaysINT8 68.3% at 91.7 ms. 60 photos cannot separate the accuracies.

**500 unseen photos** (300 stressed, 1 trial, `suite/pi_large`):

| config | accuracy | latency | vs AlwaysINT8 (gained/lost, p) |
|---|---|---|---|
| v2 / v2 + original EightSignal | 78.0% | 92.7 / 91.0 ms | 40/20, p = 0.01 |
| AlwaysINT8 | 71.3% | 101.5 ms | - |
| EightSignal cold / warm | 71.7% / 71.3% | 112.2 / 105.3 ms | not significant |
| LinUCB cold / warm, EpsilonGreedy | 73.0% / 72.7% / 71.3% | 126.8 / 132.9 / 105.1 ms | not significant |
| Baseline (FP32) | 73.3% | 130.1 ms | not significant |
| Baseline, fixed preprocessing | 85.0% | 136.5 ms | 48/7, p < 0.001 |

v2 vs EightSignal cold: 40/21, p = 0.02. v2 vs Baseline: 32/18, p = 0.06.
Most of the gain is the preprocessing fix; routing adds about 2 points at equal latency
(see the doc). v2 with the original preprocessing behaves exactly like AlwaysINT8.

## Check the preprocessing bug yourself (original code only)

Uses only the original model, label lookup and venv; no v2 code. Run it while nothing else
is running on the Pi:

```bash
cd ~/stone
venv/bin/python - <<'PY'
import os, random, numpy as np
from PIL import Image
import ai_edge_litert.interpreter as tfl
from runtime.label_lookup import build_wnid_to_label_index
m = tfl.Interpreter(model_path="models/mobilenet_v2_fp32.tflite"); m.allocate_tensors()
i, o = m.get_input_details()[0]["index"], m.get_output_details()[0]["index"]
w2i, labels = build_wnid_to_label_index("models/labels.txt")
root = "datasets/imagenette2-320/val"
files = [(os.path.join(root, w, f), w) for w in os.listdir(root) for f in os.listdir(os.path.join(root, w))]
random.Random(0).shuffle(files)
ok = {"original /255": 0, "standard /127.5-1": 0}
for p, w in files[:300]:
    a = np.array(Image.open(p).convert("RGB").resize((224, 224)))
    for name, x in [("original /255", a / 255.0), ("standard /127.5-1", a / 127.5 - 1.0)]:
        m.set_tensor(i, x[None].astype(np.float32)); m.invoke()
        ok[name] += int(np.argmax(m.get_tensor(o)) + 1 == w2i[w])
print({k: f"{100*v/300:.1f}%" for k, v in ok.items()})
PY
```

## Pipeline validation without a Pi (Apple-silicon ARM VM, tiny-imagenet)

Tiny-imagenet images are 64 px, so these runs check the code, not the claims: 30/30 unit
tests pass, v2 reproduces the original engine's outputs exactly at 224 px, `analyse_v2`
reproduces Table I from the original logs, and the online budget controller lands on its
target (2.85 ms vs 2.87 ms budget).

## Known limitations / next steps

- Slow requests get slower: escalated photos run two models (p95 about 180-190 ms vs 135 ms
  for AlwaysINT8). `BudgetController(deadline_ms=...)` can cap it; it is off by default.
- The INT8 model was probably calibrated on [0, 1] inputs too; rebuilding it needs
  `variants/build_variants.py` (internet once for the Keras ImageNet weights; tested here
  with random weights only), then re-run `pi_run_all.sh`.
- The original LinUCB/EightSignal code saves its state to disk on every request; in
  `run_suite_v2` that cost is included in their end-to-end time (as in the original).
- Frame reuse in `server_v2` is reported separately; it is never part of the protocol results.
