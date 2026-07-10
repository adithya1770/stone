import ai_edge_litert.interpreter as tflite
import numpy as np

fp32 = tflite.Interpreter(model_path="models/mobilenet_v2_fp32.tflite")
int8 = tflite.Interpreter(model_path="models/mobilenet_v2_int8.tflite")

fp32.allocate_tensors()
int8.allocate_tensors()

print("=== FP32 INPUT ===")
print(fp32.get_input_details()[0])

print("\n=== INT8 INPUT ===")
print(int8.get_input_details()[0])

print("\n=== FP32 OUTPUT ===")
print(fp32.get_output_details()[0])

print("\n=== INT8 OUTPUT ===")
print(int8.get_output_details()[0])