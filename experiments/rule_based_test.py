import time
import sys
import os
from datetime import datetime
from collections import deque
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.inference_engine import InferenceEngine
from runtime.telemetry import get_telemetry
from runtime.decision_engine import decide
from runtime.logger import initialize_logger, log_data

LOG_FILE = "logging/rule_based_log.csv"
EMA_ALPHA = 0.2
WINDOW_SIZE = 5

import argparse

parser = argparse.ArgumentParser(description="Run rule-based inference test")
parser.add_argument("--iterations", type=int, default=10, help="number of iterations to run")
args = parser.parse_args()


def adaptive_alpha(recent_values):
    if len(recent_values) < 3:
        return EMA_ALPHA
    variance = np.var(recent_values)
    if variance > 500:
        return 0.1
    elif variance > 100:
        return 0.2
    else:
        return 0.4


def update_ema(previous, current, alpha):
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous


engine = InferenceEngine(
    fp32_path="models/mobilenet_v2_fp32.tflite",
    int8_path="models/mobilenet_v2_int8.tflite",
    labels_path="models/labels.txt"
)

state = {
    "current_model": "fp32",
    "high_count": 0,
    "low_count": 0
}

initialize_logger(LOG_FILE)

ema_cpu  = None
ema_ram  = None
ema_temp = None
recent_cpu = deque(maxlen=WINDOW_SIZE)

print("Rule-based test running.")
print("-" * 50)

for iteration in range(args.iterations):
    telemetry = get_telemetry()

    recent_cpu.append(telemetry["cpu"])
    alpha = adaptive_alpha(recent_cpu)

    ema_cpu  = update_ema(ema_cpu,  telemetry["cpu"],         alpha)
    ema_ram  = update_ema(ema_ram,  telemetry["ram"],         alpha)
    ema_temp = update_ema(ema_temp, telemetry["temperature"], alpha)

    smoothed = {
        "cpu":         round(ema_cpu,  2),
        "ram":         round(ema_ram,  2),
        "temperature": round(ema_temp, 2)
    }

    decision = decide(
        smoothed["cpu"],
        smoothed["ram"],
        smoothed["temperature"],
        state
    )

    result = engine.run("dog.jpeg", decision["model"])

    log_data({
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu":          smoothed["cpu"],
        "ram":          smoothed["ram"],
        "temperature":  smoothed["temperature"],
        "health_score": decision["health_score"],
        "model":        result["model"],
        "label":        result["label"],
        "confidence":   result["confidence"],
        "latency_ms":   result["latency_ms"]
    }, LOG_FILE)

    print(f"Smoothed  : {smoothed}")
    print(f"Decision  : {decision}")
    print(f"Inference : {result}")
    print("-" * 50)

    time.sleep(1)