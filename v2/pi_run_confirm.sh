#!/usr/bin/env bash
# STONE v2 confirmation run (about 60-70 minutes): the final settings, on the same
# 800 unseen Imagewoof photos as ladder run 2.
#
#   cd ~/stone && PY=$HOME/v2env/bin/python nohup bash v2/pi_run_confirm.sh > ~/stone/v2/results_v2/confirm.log 2>&1 &
#
# Final settings: fixed time budget aimed at 145 ms (lands at about 150 on the Pi).
# Adds lite1_ladder: the best single model at about v2's average time.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1
TAG=${TAG:-confirm_$(date +%Y%m%d_%H%M)}
N=${N:-800}
TARGET=${TARGET:-145}
[ -f v2/configs/operating_points_ladder.json ] || { echo "ERROR: run pi_run_ladder.sh first (operating_points_ladder.json)"; exit 1; }
echo "STONE v2 confirmation run -> v2/results_v2/suite/$TAG  (target ${TARGET} ms; throttle flags: $(vcgencmd get_throttled 2>/dev/null || echo n/a))"
$PY -m pytest -q -p no:cacheprovider v2/tests
$PY -m v2.run_suite_v2 --trials 1 --dataset imagewoof_test --iterations "$N" --with-ladder --ladder-target-ms "$TARGET" \
    --only fixed_ladder lite1_ladder big_ladder eightsignal_ladder v2_ladder_8sig v2_ladder_cpu --tag "$TAG"
S=v2/results_v2/suite/$TAG
for ref in fixed_ladder lite1_ladder eightsignal_ladder; do
    $PY -m v2.analyse_v2 --suite "$S" --dataset imagewoof_test --ref $ref
    cp "$S/analysis/results.md" "$S/analysis/results_vs_$ref.md"
done
echo "throttle flags: $(vcgencmd get_throttled 2>/dev/null || echo n/a)"
echo "DONE -> $S/analysis/"
