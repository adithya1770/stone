#!/usr/bin/env bash
# STONE v2 FINAL run on the Raspberry Pi 4 (about 2-2.5 hours).
#
#   cd ~/stone && PY=$HOME/v2env/bin/python nohup bash v2/pi_run_final.sh > v2/results_v2/final.log 2>&1 &
#
# v2_final = the group's EightSignal controller (device state, unchanged)
#          + per-photo check (INT8 looks first, unsure photos go to FP32)
#          + per-state time budgets (idle: spare time may buy FP32; stressed: stay at INT8 cost)
#          on the rebuilt INT8 / FP32 models (v2/variants/).
#
# Re-uses the response table and latency profiles of the rebuilt-model run
# (no re-profiling), tunes operating_points_final.json from them, then runs
# his protocol and the 500-photo test. Writes ONLY into v2/results_v2/ and
# v2/configs/operating_points_final.json.
#
# Options: SRC=rebuilt_20261002_2242 (table + profiles), TRIALS=3, LARGE=500 (0 = skip),
#          SKIP_PROTOCOL=1, OLD_TABLE=... (photos excluded from the large test)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1
SRC=${SRC:-rebuilt_20261002_2242}
S=v2/results_v2/$SRC
TAG=${TAG:-final_$(date +%Y%m%d_%H%M)}
R=v2/results_v2/$TAG
TRIALS=${TRIALS:-3}
LARGE=${LARGE:-500}
OLD_TABLE=${OLD_TABLE:-v2/results_v2/pi_20261002_1807/table_imagenette.csv}
mkdir -p "$R"

for f in "$S/table_imagenette.csv" "$S/profile_idle.csv" "$S/profile_stressed.csv" \
         v2/variants/registry.json v2/variants/mnv2_fp32.tflite v2/variants/mnv2_int8.tflite \
         v2/configs/operating_points.json; do
    [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done
echo "STONE v2 FINAL on $(cat /proc/device-tree/model 2>/dev/null || uname -m) -> $R (table/profiles from $S)"

CONFIGS="baseline always_int8 linucb_cold eightsignal_cold always_int8_rebuilt baseline_rebuilt eightsignal_rebuilt v2_final"

echo; echo "[1/4] unit tests"
$PY -m pytest -q -p no:cacheprovider v2/tests

echo; echo "[2/4] tune v2_final offline (rebuilt models; idle budget = FP32 cost, stressed budget = INT8 cost)"
$PY -m v2.bench.replay_eval --table "$S/table_imagenette.csv" \
    --profile-idle "$S/profile_idle.csv" --profile-stressed "$S/profile_stressed.csv" \
    --kinds mnv2q mnv2f --budget-ref-idle mnv2f_224 --budget-ref-stressed mnv2q_224 \
    --budget-tol 0.05 --acc-tol 0.5 --label "(v2_final: rebuilt models, per-state budgets)" \
    --out-dir "$R/replay_final" --write-config --config-path v2/configs/operating_points_final.json
sleep 30

analyse () {   # suite dataset ref
    $PY -m v2.analyse_v2 --suite "$1" --dataset "$2" --ref "$3"
    cp "$1/analysis/results.md" "$1/analysis/results_vs_$3.md"
}

if [ "${SKIP_PROTOCOL:-0}" != "1" ]; then
    echo; echo "[3/4] original Pi protocol (his 100-image sequence), $TRIALS trials"
    $PY -m v2.run_suite_v2 --trials "$TRIALS" --with-final --only $CONFIGS --tag "${TAG}_protocol"
    for ref in baseline eightsignal_cold eightsignal_rebuilt; do
        analyse "v2/results_v2/suite/${TAG}_protocol" imagenette $ref
    done
fi

if [ "$LARGE" -gt 0 ]; then
    echo; echo "[4/4] large test: $LARGE unseen photos (same photos as before), stress from 40% to the end"
    EXCL="$S/table_imagenette.csv"
    [ -f "$OLD_TABLE" ] && EXCL="$EXCL $OLD_TABLE"
    $PY -m v2.run_suite_v2 --trials 1 --dataset imagenette_fresh --iterations "$LARGE" --exclude-table $EXCL \
        --with-final --only $CONFIGS --tag "${TAG}_large"
    for ref in always_int8 eightsignal_cold eightsignal_rebuilt; do
        analyse "v2/results_v2/suite/${TAG}_large" imagenette_fresh $ref
    done
fi

echo; echo "DONE. Key files:"
echo "  $R/replay_final/report.md                                    <- offline tuning of v2_final"
echo "  v2/results_v2/suite/${TAG}_large/analysis/results_vs_*.md     <- 500 photos"
echo "  v2/results_v2/suite/${TAG}_protocol/analysis/results_vs_*.md  <- his protocol"
echo "Live demo with the final system:  $PY -m v2.server_v2   (port 8001)"
