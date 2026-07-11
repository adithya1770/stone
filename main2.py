import time
import os
import csv
import itertools
from telemetry import Telemetry
from model_manager import ModelManager
from decision_engine import DecisionEngine
from logger import Logger

ACTION_TO_MODEL_TYPE = {0: "fp32", 1: "int8"}
TEST_IMAGES_DIR = "test_images"

def load_ground_truth(csv_path):
    gt = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            gt[row["filename"]] = row["ground_truth_label"]
    return gt

def compute_reward(predicted_label, ground_truth_label, latency_ms):
    correct = 1.0 if predicted_label == ground_truth_label else 0.0
    return correct - (0.01 * latency_ms)

ground_truth = load_ground_truth(os.path.join(TEST_IMAGES_DIR, "ground_truth.csv"))
image_cycle = itertools.cycle(sorted(ground_truth.keys()))

telemetry = Telemetry()
mm = ModelManager("models/mobilenet_v2_fp32.tflite", "models/mobilenet_v2_int8.tflite", "models/labels.txt")
engine = DecisionEngine(min_mode_duration=5)
logger = Logger("adaptive_run.csv")

input_id = 0
try:
    while True:
        before_snap = telemetry.read()
        action = engine.decide(before_snap)
        model_type = ACTION_TO_MODEL_TYPE[action]

        image_filename = next(image_cycle)
        image_path = os.path.join(TEST_IMAGES_DIR, image_filename)
        gt_label = ground_truth[image_filename]

        result = mm.predict(image_path, model_type)
        after_snap = telemetry.read()
        reward = compute_reward(result["label"], gt_label, result["latency_ms"])

        logger.log(before_snap, after_snap, input_id, action, model_type,
                    result, ground_truth=gt_label, reward=reward)

        print(before_snap, result, "gt:", gt_label, "reward:", round(reward, 4))
        input_id += 1
        time.sleep(3)
except KeyboardInterrupt:
    logger.close()
    print("Stopped, log saved.")