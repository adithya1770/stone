import time
import threading
import argparse
from datetime import datetime
from collections import deque
import numpy as np

from runtime.inference_engine import InferenceEngine
from runtime.telemetry import get_telemetry
from runtime.decision_engine import get_algo, choose, update, get_last_source, set_extra_telemetry
from runtime.reward import calculate_reward
from runtime.logger import initialize_logger, log_data
from runtime.image_pool import ImagePool
from runtime.label_lookup import build_wnid_to_label_index, true_label_for_image


parser = argparse.ArgumentParser(description="Stone Adaptive Runtime")

parser.add_argument(
    "--algo",
    type=str,
    default="linucb",
    choices=["linucb", "egreedy", "eightsignal"],
    help="Decision algorithm"
)

parser.add_argument(
    "--log",
    type=str,
    default="logging/linucb_log.csv",
    help="Log file path"
)

parser.add_argument(
    "--iterations",
    type=int,
    default=100,
    help="Number of iterations"
)

args = parser.parse_args()

DECISION_ALGO = args.algo
LOG_FILE = args.log
MAX_ITERATIONS = args.iterations

EMA_ALPHA = 0.2
WINDOW_SIZE = 5


_telemetry_state = {
    "cpu": 0.0,
    "ram": 0.0,
    "temperature": 40.0,
    "battery": 100.0,
    "cpu_per_core": []
}

_telemetry_lock = threading.Lock()


def _telemetry_worker():

    while True:

        fresh = get_telemetry()

        with _telemetry_lock:

            _telemetry_state["cpu"] = fresh["cpu"]
            _telemetry_state["ram"] = fresh["ram"]
            _telemetry_state["temperature"] = fresh["temperature"]
            _telemetry_state["battery"] = fresh["battery"]
            _telemetry_state["cpu_per_core"] = fresh["cpu_per_core"]
            _telemetry_state["cpu_freq_current"] = fresh["cpu_freq_current"]
            _telemetry_state["cpu_freq_max"] = fresh["cpu_freq_max"]
            _telemetry_state["disk_busy_time"] = fresh["disk_busy_time"]


def read_telemetry():

    with _telemetry_lock:

        return dict(_telemetry_state)


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


_t = threading.Thread(
    target=_telemetry_worker,
    daemon=True
)

_t.start()

print("Telemetry thread started.")

time.sleep(0.6)

engine = InferenceEngine(
    fp32_path="models/mobilenet_v2_fp32.tflite",
    int8_path="models/mobilenet_v2_int8.tflite",
    labels_path="models/labels.txt"
)

import os as _os
seed = int(_os.environ.get("STONE_SEED", 42))
image_pool = ImagePool(seed=seed)
wnid_to_label_index, labels = build_wnid_to_label_index("models/labels.txt")

state = {
    "current_model": "fp32",
    "high_count": 0,
    "low_count": 0
}

algo = get_algo(
    DECISION_ALGO,
    alpha=1.0
)

initialize_logger(LOG_FILE)

ema_cpu = None
ema_ram = None
ema_temp = None

recent_cpu = deque(maxlen=WINDOW_SIZE)

print(f"Stone adaptive runtime — {DECISION_ALGO.upper()} mode.")
print(f"Logging to: {LOG_FILE}")
print("-" * 50)
for iteration in range(MAX_ITERATIONS):

    telemetry = read_telemetry()

    set_extra_telemetry(
        algo,
        telemetry.get("cpu_freq_current", 0.0),
        telemetry.get("cpu_freq_max", 4000.0),
        telemetry.get("disk_busy_time", 0)
    )

    recent_cpu.append(telemetry["cpu"])
    alpha = adaptive_alpha(recent_cpu)

    ema_cpu = update_ema(
        ema_cpu,
        telemetry["cpu"],
        alpha
    )

    ema_ram = update_ema(
        ema_ram,
        telemetry["ram"],
        alpha
    )

    ema_temp = update_ema(
        ema_temp,
        telemetry["temperature"],
        alpha
    )

    smoothed = {
        "cpu": round(ema_cpu, 2),
        "ram": round(ema_ram, 2),
        "temperature": round(ema_temp, 2),
        "battery": telemetry["battery"]
    }

    decision_start = time.time()

    chosen_model, scores = choose(
        algo,
        smoothed["cpu"],
        smoothed["ram"],
        smoothed["temperature"],
        smoothed["battery"]
    )

    decision_source = get_last_source(algo) or "algo"

    decision_time_ms = round((time.time() - decision_start) * 1000, 3)

    image_path = image_pool.get(iteration)
    true_label = true_label_for_image(image_path, wnid_to_label_index, labels)

    result = engine.run(
        image_path,
        chosen_model
    )

    is_correct = (
        result["label"].strip().lower() == true_label.strip().lower()
        if true_label else None
    )

    reward = calculate_reward(
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        cpu=smoothed["cpu"],
        ram=smoothed["ram"],
        temperature=smoothed["temperature"],
        decision_time_ms=decision_time_ms
    )

    update(
        algo,
        chosen_model,
        smoothed["cpu"],
        smoothed["ram"],
        smoothed["temperature"],
        smoothed["battery"],
        reward,
        confidence=result["confidence"]
    )

    log_data({
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "cpu": smoothed["cpu"],
        "ram": smoothed["ram"],
        "temperature": smoothed["temperature"],
        "battery": smoothed["battery"],
        "health_score": round(scores["fp32"], 4),
        "model": result["model"],
        "decision_source": decision_source,
        "image_path": image_path,
        "true_label": true_label if true_label else "unmatched",
        "label": result["label"],
        "correct": is_correct,
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
        "decision_time_ms": decision_time_ms
    }, LOG_FILE)

    print("Raw       :", telemetry)
    print("Per-core  :", telemetry["cpu_per_core"])
    print("Alpha     :", alpha)
    print("Smoothed  :", smoothed)
    print(
        "Scores    :",
        {k: round(v, 4) for k, v in scores.items()}
    )
    print("Chosen    :", chosen_model)
    print("Decision  :", f"{decision_time_ms}ms")
    print("Reward    :", reward)
    print("Image     :", image_path)
    print("True label:", true_label)
    print("Inference :", result)
    print("Correct   :", is_correct)
    print("Source    :", decision_source)
    print("-" * 50)

    time.sleep(1)

print()
print("=" * 50)
print("Experiment completed.")
print("=" * 50)