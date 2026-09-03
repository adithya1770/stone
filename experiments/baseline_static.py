import time
import sys
import os
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.inference_engine import InferenceEngine
from runtime.telemetry import get_telemetry
from runtime.logger import initialize_logger, log_data
from runtime.image_pool import ImagePool
from runtime.label_lookup import build_wnid_to_label_index, true_label_for_image

LOG_FILE = "logging/baseline_log.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
args = parser.parse_args()

engine = InferenceEngine(
    fp32_path="models/mobilenet_v2_fp32.tflite",
    int8_path="models/mobilenet_v2_int8.tflite",
    labels_path="models/labels.txt"
)

import os as _os
seed = int(_os.environ.get("STONE_SEED", 42))
image_pool = ImagePool(seed=seed)
wnid_to_label_index, labels = build_wnid_to_label_index("models/labels.txt")

initialize_logger(LOG_FILE)

print("Baseline running — always FP32, no switching.")
print("-" * 50)

for iteration in range(args.iterations):
    telemetry = get_telemetry()
    image_path = image_pool.get(iteration)
    true_label = true_label_for_image(image_path, wnid_to_label_index, labels)

    result = engine.run(image_path, "fp32")

    is_correct = (
        result["label"].strip().lower() == true_label.strip().lower()
        if true_label else None
    )

    log_data({
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu":          telemetry["cpu"],
        "ram":          telemetry["ram"],
        "temperature":  telemetry["temperature"],
        "battery":      telemetry["battery"],
        "health_score": 0,
        "model":        result["model"],
        "image_path":   image_path,
        "true_label":   true_label if true_label else "unmatched",
        "label":        result["label"],
        "correct":      is_correct,
        "confidence":   result["confidence"],
        "latency_ms":   result["latency_ms"],
        "decision_time_ms": 0
    }, LOG_FILE)

    print(f"Telemetry : {telemetry}")
    print(f"Image     : {image_path}")
    print(f"True label: {true_label}")
    print(f"Inference : {result}")
    print(f"Correct   : {is_correct}")
    print("-" * 50)

    time.sleep(1)