from runtime.inference_engine import InferenceEngine

engine = InferenceEngine(
    fp32_path="models/mobilenet_v2_fp32.tflite",
    int8_path="models/mobilenet_v2_int8.tflite",
    labels_path="models/labels.txt"
)

fp32_result = engine.run("dog.jpeg", "fp32")
int8_result = engine.run("dog.jpeg", "int8")

print("FP32 result:", fp32_result)
print("INT8 result:", int8_result)