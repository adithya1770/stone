import time
from datetime import datetime

from runtime.inference_engine import InferenceEngine
from runtime.telemetry import get_telemetry
from runtime.decision_engine import decide
from runtime.logger import initialize_logger, log_data
from runtime.reward import calculate_reward

r = calculate_reward(0.55, 13)
print("Reward (FP32 healthy):", r)

r = calculate_reward(0.55, 38)
print("Reward (FP32 stressed):", r)

r = calculate_reward(0.24, 12)
print("Reward (INT8 stressed):", r)


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

initialize_logger()

while True:
    telemetry = get_telemetry()

    decision = decide(
        telemetry["cpu"],
        telemetry["ram"],
        telemetry["temperature"],
        state
    )

    result = engine.run("dog.jpeg", decision["model"])

    log_data({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": telemetry["cpu"],
        "ram": telemetry["ram"],
        "temperature": telemetry["temperature"],
        "health_score": decision["health_score"],
        "model": result["model"],
        "label": result["label"],
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"]
    })

    print("-" * 50)
    print("Telemetry :", telemetry)
    print("Decision  :", decision)
    print("Inference :", result)

    time.sleep(1)