import time
from telemetry import Telemetry
from model_manager import ModelManager
from decision_engine import DecisionEngine

ACTION_TO_MODEL_TYPE = {0: "fp32", 1: "int8"}

telemetry = Telemetry()
mm = ModelManager(
    "models/mobilenet_v2_fp32.tflite",
    "models/mobilenet_v2_int8.tflite",
    "models/labels.txt"
)
engine = DecisionEngine(min_mode_duration=5)

while True:
    snap = telemetry.read()
    action = engine.decide(snap)
    model_type = ACTION_TO_MODEL_TYPE[action]
    result = mm.predict("dog.jpeg", model_type)
    print("\nTelemetry:", snap)
    print("Chosen:", model_type)
    print("Result:", result)
    time.sleep(3)
