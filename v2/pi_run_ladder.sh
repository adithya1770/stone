#!/usr/bin/env bash
# STONE v2 LADDER run on the Raspberry Pi 4 (about 75-90 minutes).
#
#   cd ~/stone && PY=$HOME/v2env/bin/python nohup bash v2/pi_run_ladder.sh > v2/results_v2/ladder.log 2>&1 &
#
# Models: MobileNetV2 (rebuilt INT8) + official EfficientNet-Lite0..4 INT8 (v2/models_ext).
# Task:   Imagewoof (10 dog breeds). Thresholds are tuned on one half of Imagewoof val
#         (accuracy table measured on the Mac, latencies from v2/pi_profile_ladder.sh),
#         the live run uses photos from the OTHER half only.
# Goal:   best accuracy while keeping the average time per photo within BUDGET ms,
#         both when the Pi is idle and when it is stressed.
# Configs (all in the same session, rotated order, cool-down between them):
#   baseline, always_int8, eightsignal_cold  - the original system (his models)
#   fixed_ladder        one model that fits the budget even under stress
#   big_ladder          the best model for an idle Pi, kept under stress (breaks the budget)
#   eightsignal_ladder  EightSignal switches between the best idle model and the best stressed model
#   v2_ladder           EightSignal (device state) + per-photo check (small model first,
#                       unsure photos go to a bigger one) + per-state time budgets
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1
TAG=${TAG:-ladder_$(date +%Y%m%d_%H%M)}
R=v2/results_v2/$TAG
BUDGET=${BUDGET:-150}
N=${N:-800}
TABLE=v2/data/ladder/table_imagewoof.csv
PROF=v2/results_v2/ladder_pi
mkdir -p "$R"
for f in "$TABLE" "$PROF/profile_idle.csv" "$PROF/profile_stressed.csv" datasets/imagewoof2-320/val \
         v2/models_ext/efficientnet-lite4-int8.tflite v2/variants/mnv2_int8.tflite v2/configs/operating_points.json; do
    [ -e "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done
echo "STONE v2 LADDER on $(cat /proc/device-tree/model 2>/dev/null || uname -m) -> $R  (budget ${BUDGET} ms, ${N} photos)"
echo "throttle flags: $(vcgencmd get_throttled 2>/dev/null || echo n/a)"

echo; echo "[1/4] unit tests"
$PY -m pytest -q -p no:cacheprovider v2/tests

echo; echo "[2/4] tune on the Imagewoof tune half (budget ${BUDGET} ms in every state)"
$PY -m v2.bench.replay_eval --table "$TABLE" \
    --profile-idle "$PROF/profile_idle.csv" --profile-stressed "$PROF/profile_stressed.csv" \
    --kinds mnv2q efl0q efl1q efl2q efl3q efl4q --budget-idle "$BUDGET" --budget-stressed "$BUDGET" \
    --acc-tol 0.3 --label "(ladder, ${BUDGET} ms budget)" --out-dir "$R/replay_ladder" \
    --write-config --config-path v2/configs/operating_points_ladder.json
$PY -c "import json; r=json.load(open('v2/configs/operating_points_ladder.json')); print('reference policies:', r['_reference']); print({k: (v['first'], v.get('second')) for k, v in r.items() if not k.startswith('_')})"
sleep 30

echo; echo "[3/4] live run: ${N} photos from the Imagewoof test half, stress from 40% to the end"
$PY -m v2.run_suite_v2 --trials 1 --dataset imagewoof_test --iterations "$N" --with-ladder \
    --only baseline always_int8 eightsignal_cold fixed_ladder big_ladder eightsignal_ladder v2_ladder --tag "$TAG"

echo; echo "[4/4] analysis"
S=v2/results_v2/suite/$TAG
for ref in fixed_ladder eightsignal_ladder eightsignal_cold; do
    $PY -m v2.analyse_v2 --suite "$S" --dataset imagewoof_test --ref $ref
    cp "$S/analysis/results.md" "$S/analysis/results_vs_$ref.md"
done
echo "throttle flags: $(vcgencmd get_throttled 2>/dev/null || echo n/a)"
echo; echo "DONE -> $S/analysis/results_vs_*.md   and   $R/replay_ladder/report.md"
