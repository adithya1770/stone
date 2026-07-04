import time

from runtime.inference_engine import InferenceEngine
from runtime.telemetry import get_telemetry
from runtime.decision_engine import decide


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

while True:

    telemetry = get_telemetry()

    decision = decide(
        telemetry["cpu"],
        telemetry["ram"],
        telemetry["temperature"],
        state
    )

    result = engine.run(
        "dog.jpeg",
        decision["model"]
    )

    print("-" * 50)
    print("Telemetry :", telemetry)
    print("Decision  :", decision)
    print("Inference :", result)

    time.sleep(1)