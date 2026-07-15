import time
import sys
import os
import argparse
from datetime import datetime


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.inference_engine import InferenceEngine
from runtime.telemetry import get_telemetry
from runtime.logger import initialize_logger, log_data

LOG_FILE = "logging/baseline_log.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
args = parser.parse_args()

engine = InferenceEngine(
    fp32_path="models/mobilenet_v2_fp32.tflite",
    int8_path="models/mobilenet_v2_int8.tflite",
    labels_path="models/labels.txt"
)

initialize_logger(LOG_FILE)

print("Baseline running — always FP32, no switching.")
print("-" * 50)

for iteration in range(args.iterations):
    telemetry = get_telemetry()
    result = engine.run("dog.jpeg", "fp32")

    log_data({
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu":          telemetry["cpu"],
        "ram":          telemetry["ram"],
        "temperature":  telemetry["temperature"],
        "health_score": 0,
        "model":        result["model"],
        "label":        result["label"],
        "confidence":   result["confidence"],
        "latency_ms":   result["latency_ms"]
    }, LOG_FILE)

    print(f"Telemetry : {telemetry}")
    print(f"Inference : {result}")
    print("-" * 50)

    time.sleep(1)