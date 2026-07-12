import time
import threading
import argparse
from datetime import datetime
from collections import deque
import numpy as np

from runtime.inference_engine import InferenceEngine
from runtime.telemetry import get_telemetry
from runtime.decision_engine import decide, get_algo
from runtime.reward import calculate_reward
from runtime.logger import initialize_logger, log_data


parser = argparse.ArgumentParser(description="Stone Adaptive Runtime")
parser.add_argument(
    "--algo",
    type=str,
    default="linucb",
    choices=["linucb", "thompson"],
    help="Decision algorithm: linucb or thompson"
)
parser.add_argument(
    "--log",
    type=str,
    default="logging/linucb_log.csv",
    help="Log file path"
)
args = parser.parse_args()

DECISION_ALGO = args.algo
LOG_FILE      = args.log
EMA_ALPHA     = 0.2
WINDOW_SIZE   = 5


_telemetry_state = {
    "cpu":          0.0,
    "ram":          0.0,
    "temperature":  40.0,
    "cpu_per_core": []
}
_telemetry_lock = threading.Lock()


def _telemetry_worker():
    while True:
        fresh = get_telemetry()
        with _telemetry_lock:
            _telemetry_state["cpu"]          = fresh["cpu"]
            _telemetry_state["ram"]          = fresh["ram"]
            _telemetry_state["temperature"]  = fresh["temperature"]
            _telemetry_state["cpu_per_core"] = fresh["cpu_per_core"]


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


_t = threading.Thread(target=_telemetry_worker, daemon=True)
_t.start()
print("Telemetry thread started.")
time.sleep(0.6)

engine = InferenceEngine(
    fp32_path="models/mobilenet_v2_fp32.tflite",
    int8_path="models/mobilenet_v2_int8.tflite",
    labels_path="models/labels.txt"
)

state = {
    "current_model": "fp32",
    "high_count":    0,
    "low_count":     0
}

algo = get_algo(DECISION_ALGO, alpha=1.0)
initialize_logger(LOG_FILE)

ema_cpu    = None
ema_ram    = None
ema_temp   = None
recent_cpu = deque(maxlen=WINDOW_SIZE)

print(f"Stone adaptive runtime — {DECISION_ALGO.upper()} mode.")
print(f"Logging to: {LOG_FILE}")
print("-" * 50)

while True:
    telemetry = read_telemetry()

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

    chosen_model, scores = algo.choose(
        smoothed["cpu"],
        smoothed["ram"],
        smoothed["temperature"]
    )

    result = engine.run("dog.jpeg", chosen_model)

    reward = calculate_reward(result["confidence"], result["latency_ms"])

    algo.update(
        chosen_model,
        smoothed["cpu"],
        smoothed["ram"],
        smoothed["temperature"],
        reward
    )

    log_data({
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu":          smoothed["cpu"],
        "ram":          smoothed["ram"],
        "temperature":  smoothed["temperature"],
        "health_score": round(scores["fp32"], 4),
        "model":        result["model"],
        "label":        result["label"],
        "confidence":   result["confidence"],
        "latency_ms":   result["latency_ms"]
    }, LOG_FILE)

    print("Raw       :", telemetry)
    print("Per-core  :", telemetry["cpu_per_core"])
    print("Alpha     :", alpha)
    print("Smoothed  :", smoothed)
    print("Scores    :", {k: round(v, 4) for k, v in scores.items()})
    print("Chosen    :", chosen_model)
    print("Reward    :", reward)
    print("Inference :", result)
    print("-" * 50)

    time.sleep(1)