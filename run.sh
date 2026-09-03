#!/bin/bash

set -e

echo "========================================="
echo "STONE Experimental Evaluation — Final 5"
echo "========================================="

ITERATIONS=100

run_dynamic_workload() {

    (
        echo "[Workload] Idle (35s)"
        sleep 35

        echo "[Workload] Stress (35s)"
        stress-ng --cpu 0 --timeout 35s > /dev/null 2>&1

        echo "[Workload] Recovery (30s)"
        sleep 30

        echo "[Workload] Finished"
    ) &

    WORKLOAD_PID=$!
}


run_experiment() {

    NAME=$1
    COMMAND=$2

    echo
    echo "========================================="
    echo "Running $NAME"
    echo "========================================="

    run_dynamic_workload

    eval "$COMMAND"

    wait $WORKLOAD_PID

    echo
    echo "$NAME completed."
}


echo
echo "Cleaning previous logs..."

rm -f logging/*.csv
rm -f logging/*.npz

echo "Done."

########################################################
# Baseline (always FP32)
########################################################

run_experiment \
"Baseline" \
"python3 experiments/baseline_static.py --iterations $ITERATIONS"

########################################################
# Always INT8
########################################################

run_experiment \
"Always INT8" \
"python3 experiments/always_int8_static.py --iterations $ITERATIONS"

########################################################
# LinUCB Cold
# (state file wiped just before this run, so it learns from scratch)
########################################################

rm -f logging/linucb_state.npz

run_experiment \
"LinUCB Cold" \
"python3 main.py \
--algo linucb \
--iterations $ITERATIONS \
--log logging/linucb_cold_log.csv"

########################################################
# LinUCB Warm
# (deliberately NOT deleting the state file here — this run
# continues learning from whatever LinUCB Cold just built up)
########################################################

run_experiment \
"LinUCB Warm" \
"python3 main.py \
--algo linucb \
--iterations $ITERATIONS \
--log logging/linucb_warm_log.csv"

########################################################
# Eight Signal Cold
# (its internal LinUCB fallback state file wiped, so that
# component learns from scratch — the 8-signal voting logic
# itself has no persistence and is always "fresh" regardless)
########################################################

rm -f logging/eight_signal_linucb_state.npz

run_experiment \
"Eight Signal Cold" \
"python3 main.py \
--algo eightsignal \
--iterations $ITERATIONS \
--log logging/eightsignal_cold_log.csv"

########################################################
# Eight Signal Warm
# (deliberately NOT deleting eight_signal_linucb_state.npz —
# this run's internal LinUCB fallback continues learning from
# whatever Eight Signal Cold just built up)
########################################################

run_experiment \
"Eight Signal Warm" \
"python3 main.py \
--algo eightsignal \
--iterations $ITERATIONS \
--log logging/eightsignal_warm_log.csv"

########################################################
# Results
########################################################

echo
echo "========================================="
echo "Running Analysis"
echo "========================================="

python3 experiments/analyse_results.py

echo
echo "========================================="
echo "ALL EXPERIMENTS FINISHED"
echo "========================================="
echo
echo "Logs saved in : logging/"
echo