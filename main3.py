import time
import os
import csv
import itertools
from telemetry import Telemetry
from model_manager import ModelManager
from decision_engine import DecisionEngine
from logger import Logger
from linucb import LinUCB
from context import N_FEATURES
from bootstrap_linucb import bootstrap_from_csv

ACTION_TO_MODEL_TYPE = {0: "fp32", 1: "int8"}
TEST_IMAGES_DIR = "test_images"
BOOTSTRAP_CSV = "adaptive_run.csv"   # the heuristic-phase log, used only to seed LinUCB
LIVE_LOG_CSV = "linucb_run.csv"      # new file for this LinUCB-driven run — keeps
                                       # bootstrap data separate from live data, so you
                                       # can always tell which rows came from which phase


def load_ground_truth(csv_path):
    gt = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            gt[row["filename"]] = row["ground_truth_label"]
    return gt


def compute_reward(predicted_label, ground_truth_label, latency_ms):
    correct = 1.0 if predicted_label == ground_truth_label else 0.0
    return correct - (0.01 * latency_ms)


# --- Set up LinUCB and bootstrap it from the heuristic-phase log ---
linucb = LinUCB(n_actions=2, n_features=N_FEATURES, alpha=1.0)
n_replayed = bootstrap_from_csv(linucb, BOOTSTRAP_CSV)
print(f"Bootstrapped LinUCB with {n_replayed} historical rows from {BOOTSTRAP_CSV}")
print("Starting theta (FP32 arm):", linucb.theta(0))
print("Starting theta (INT8 arm):", linucb.theta(1))

# --- Set up the rest of the pipeline, same as before ---
ground_truth = load_ground_truth(os.path.join(TEST_IMAGES_DIR, "ground_truth.csv"))
image_cycle = itertools.cycle(sorted(ground_truth.keys()))

telemetry = Telemetry()
mm = ModelManager("models/mobilenet_v2_fp32.tflite", "models/mobilenet_v2_int8.tflite", "models/labels.txt")
engine = DecisionEngine(linucb, min_mode_duration=5)
logger = Logger(LIVE_LOG_CSV)

input_id = 0
try:
    while True:
        before_snap = telemetry.read()
        action, context = engine.decide(before_snap)   # note: now returns (action, context)
        model_type = ACTION_TO_MODEL_TYPE[action]

        image_filename = next(image_cycle)
        image_path = os.path.join(TEST_IMAGES_DIR, image_filename)
        gt_label = ground_truth[image_filename]

        result = mm.predict(image_path, model_type)
        after_snap = telemetry.read()
        reward = compute_reward(result["label"], gt_label, result["latency_ms"])

        logger.log(before_snap, after_snap, input_id, action, model_type,
                    result, ground_truth=gt_label, reward=reward)

        engine.learn(action, context, reward, result["latency_ms"])  # LinUCB updates here

        print(before_snap, result, "gt:", gt_label, "reward:", round(reward, 4))
        input_id += 1
        time.sleep(3)
except KeyboardInterrupt:
    logger.close()
    print("Stopped, log saved.")