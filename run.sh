#!/bin/bash

set -e

echo "========================================="
echo "STONE Experimental Evaluation"
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
# Baseline
########################################################

run_experiment \
"Baseline" \
"python3 experiments/baseline_static.py --iterations $ITERATIONS"

########################################################
# Rule-Based
########################################################

run_experiment \
"Rule-Based" \
"python3 experiments/rule_based_test.py --iterations $ITERATIONS"

########################################################
# Epsilon Greedy
########################################################

run_experiment \
"Epsilon Greedy" \
"python3 main.py \
--algo egreedy \
--iterations $ITERATIONS \
--log logging/egreedy_log.csv"

########################################################
# Thompson Sampling
########################################################

run_experiment \
"Thompson Sampling" \
"python3 main.py \
--algo thompson \
--iterations $ITERATIONS \
--log logging/thompson_log.csv"

########################################################
# LinUCB Cold
########################################################

rm -f logging/linucb_state.npz

run_experiment \
"LinUCB Cold" \
"python3 main.py \
--algo linucb \
--iterations $ITERATIONS \
--log logging/linucb_log.csv"

########################################################
# LinUCB Warm
########################################################

run_experiment \
"LinUCB Warm" \
"python3 main.py \
--algo linucb \
--iterations $ITERATIONS \
--log logging/linucb_warm_log.csv"

########################################################
# Sliding LinUCB Cold
########################################################

rm -f logging/sliding_linucb_state.npz

run_experiment \
"Sliding LinUCB Cold" \
"python3 main.py \
--algo sliding \
--iterations $ITERATIONS \
--log logging/sliding_log_cold.csv"

########################################################
# Sliding LinUCB Warm
########################################################

run_experiment \
"Sliding LinUCB Warm" \
"python3 main.py \
--algo sliding \
--iterations $ITERATIONS \
--log logging/sliding_log_warm.csv"

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