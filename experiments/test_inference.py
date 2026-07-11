from model_manager import ModelManager

manager = ModelManager(
    "models/mobilenet_v2_fp32.tflite",
    "models/mobilenet_v2_int8.tflite",
    "models/labels.txt"
)

print(manager.predict(
    "cat.jpeg",
    "fp32"
))

print(manager.predict(
    "cat.jpeg",
    "int8"
))