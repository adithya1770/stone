#!/usr/bin/env bash
# STONE v2 — INT8 + FP32 run with the REBUILT models (about 2-2.5 hours).
#
#   cd ~/stone && nohup bash v2/pi_run_rebuilt.sh > v2/results_v2/rebuilt.log 2>&1 &
#
# Needs v2/variants/mnv2_fp32.tflite, mnv2_int8.tflite and registry.json
# (built by v2/variants/build_variants.py from the official Keras weights:
# true FP32 + full-integer INT8 calibrated with the correct input range).
# Writes ONLY into v2/results_v2/ and v2/configs/operating_points_rebuilt.json.
# Existing results and operating points are not touched.
#
# Options (environment variables):
#   TRIALS=3            trials of the original 100-image protocol
#   LIMIT=800           tuning images (same 800 as the first Pi run)
#   LARGE=500           photos in the large test (0 = skip)
#   SKIP_PROTOCOL=1     skip the 100-image protocol
#   OLD_TABLE=...       first run's response table (its photos are excluded from the large test)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
export PYTHONDONTWRITEBYTECODE=1
TAG=${TAG:-rebuilt_$(date +%Y%m%d_%H%M)}
R=v2/results_v2/$TAG
TRIALS=${TRIALS:-3}
LIMIT=${LIMIT:-800}
LARGE=${LARGE:-500}
OLD_TABLE=${OLD_TABLE:-v2/results_v2/pi_20261002_1807/table_imagenette.csv}
mkdir -p "$R"

[ -d datasets/imagenette2-320/val ] || { echo "ERROR: datasets/imagenette2-320/val not found"; exit 1; }
for f in v2/variants/registry.json v2/variants/mnv2_fp32.tflite v2/variants/mnv2_int8.tflite \
         v2/configs/operating_points.json v2/configs/operating_points_fixed.json; do
    [ -f "$f" ] || { echo "ERROR: $f missing"; exit 1; }
done
$PY -c "import json; r=json.load(open('v2/variants/registry.json')); assert 'mnv2q' in r and 'mnv2f' in r, r" \
    || { echo "ERROR: registry.json must list mnv2q and mnv2f"; exit 1; }
echo "STONE v2 rebuilt-model run on $(cat /proc/device-tree/model 2>/dev/null || uname -m) -> $R"

CONFIGS="baseline always_int8 eightsignal_cold baseline_fixed v2_resource_aware_fixed always_int8_rebuilt baseline_rebuilt v2_rebuilt"

echo; echo "[1/7] unit tests"
$PY -m pytest -q -p no:cacheprovider v2/tests

echo; echo "[2/7] latency profile: idle (0.5 s gaps), original + rebuilt variants"
$PY -m v2.bench.profile_variants --condition idle --runs 40 --gap 0.5 --out "$R/profile_idle.csv"
sleep 60
echo; echo "[3/7] latency profile: stressed (stress-ng --cpu 0)"
$PY -m v2.bench.profile_variants --condition stressed --runs 30 --gap 0.5 --out "$R/profile_stressed.csv"
sleep 60

echo; echo "[4/7] response table: every variant x $LIMIT held-out imagenette images (same images as the first run)"
$PY -m v2.bench.build_response_table --dataset imagenette --exclude-his-pool --limit "$LIMIT" --out "$R/table_imagenette.csv"

echo; echo "[5/7] offline replay"
# v2_rebuilt may only use the two rebuilt models (INT8 + FP32) at any resolution;
# his Baseline and AlwaysINT8 stay in the report as reference rows.
$PY -m v2.bench.replay_eval --table "$R/table_imagenette.csv" \
    --profile-idle "$R/profile_idle.csv" --profile-stressed "$R/profile_stressed.csv" \
    --kinds mnv2q mnv2f --label "(rebuilt INT8 + FP32)" --out-dir "$R/replay_rebuilt" \
    --write-config --config-path v2/configs/operating_points_rebuilt.json
# report only (no config written): every model kind together
$PY -m v2.bench.replay_eval --table "$R/table_imagenette.csv" \
    --profile-idle "$R/profile_idle.csv" --profile-stressed "$R/profile_stressed.csv" \
    --label "(all models)" --out-dir "$R/replay_all"
sleep 60

analyse () {   # suite dataset ref
    $PY -m v2.analyse_v2 --suite "$1" --dataset "$2" --ref "$3"
    cp "$1/analysis/results.md" "$1/analysis/results_vs_$3.md"
}

if [ "${SKIP_PROTOCOL:-0}" != "1" ]; then
    echo; echo "[6/7] original Pi protocol (his 100-image sequence), $TRIALS trials"
    $PY -m v2.run_suite_v2 --trials "$TRIALS" --with-fixed --with-rebuilt --only $CONFIGS --tag "${TAG}_protocol"
    analyse "v2/results_v2/suite/${TAG}_protocol" imagenette baseline
    analyse "v2/results_v2/suite/${TAG}_protocol" imagenette always_int8
fi

if [ "$LARGE" -gt 0 ]; then
    echo; echo "[7/7] large test: $LARGE unseen photos (same photos as suite/pi_large), stress from 40% to the end"
    EXCL="$R/table_imagenette.csv"
    [ -f "$OLD_TABLE" ] && EXCL="$EXCL $OLD_TABLE"
    $PY -m v2.run_suite_v2 --trials 1 --dataset imagenette_fresh --iterations "$LARGE" --exclude-table $EXCL \
        --with-fixed --with-rebuilt --only $CONFIGS --tag "${TAG}_large"
    analyse "v2/results_v2/suite/${TAG}_large" imagenette_fresh always_int8
    analyse "v2/results_v2/suite/${TAG}_large" imagenette_fresh always_int8_rebuilt
    analyse "v2/results_v2/suite/${TAG}_large" imagenette_fresh baseline_rebuilt
fi

echo; echo "DONE. Key files:"
echo "  $R/replay_rebuilt/report.md                                  <- offline: rebuilt INT8 + FP32"
echo "  v2/results_v2/suite/${TAG}_large/analysis/results_vs_*.md    <- 500 photos"
echo "  v2/results_v2/suite/${TAG}_protocol/analysis/results_vs_*.md <- his protocol"
