from telemetry import Telemetry
from decision_engine import (
    DecisionEngine,
    ACTION_FP32,
    ACTION_INT8
)
from model_manager import ModelManager


monitor = Telemetry()

engine = DecisionEngine()

manager = ModelManager(
    "models/mobilenet_v2_fp32.tflite",
    "models/mobilenet_v2_int8.tflite",
    "models/labels.txt"
)

telemetry = monitor.read()

action = engine.choose_action(
    telemetry
)

model_type = (
    "fp32"
    if action == ACTION_FP32
    else "int8"
)

result = manager.predict(
    "cat.jpeg",
    model_type
)

print("\nTelemetry")
print(telemetry)

print("\nDecision")
print(model_type)

print("\nInference")
print(result)