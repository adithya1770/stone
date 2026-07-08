import time
from datetime import datetime

from runtime.inference_engine import InferenceEngine
from runtime.telemetry import get_telemetry
from runtime.decision_engine import decide, LinUCB
from runtime.reward import calculate_reward
from runtime.logger import initialize_logger, log_data


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

linucb = LinUCB(alpha=1.0)
initialize_logger()

while True:
    telemetry = get_telemetry()

    chosen_model, scores = linucb.choose(
        telemetry["cpu"],
        telemetry["ram"],
        telemetry["temperature"]
    )

    result = engine.run("dog.jpeg", chosen_model)

    reward = calculate_reward(result["confidence"], result["latency_ms"])

    linucb.update(
        chosen_model,
        telemetry["cpu"],
        telemetry["ram"],
        telemetry["temperature"],
        reward
    )

    log_data({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": telemetry["cpu"],
        "ram": telemetry["ram"],
        "temperature": telemetry["temperature"],
        "health_score": round(scores["fp32"], 4),
        "model": result["model"],
        "label": result["label"],
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"]
    })

    print("-" * 50)
    print("Telemetry :", telemetry)
    print("Scores    :", {k: round(v, 4) for k, v in scores.items()})
    print("Chosen    :", chosen_model)
    print("Reward    :", reward)
    print("Inference :", result)

    time.sleep(1)