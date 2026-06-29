import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ---------- LOAD MODELS ----------

fp32 = tf.lite.Interpreter(
    model_path="models/mobilenet_v2_fp32.tflite"
)

int8 = tf.lite.Interpreter(
    model_path="models/mobilenet_v2_int8.tflite"
)

fp32.allocate_tensors()
int8.allocate_tensors()

# ---------- LOAD IMAGE ----------

img = Image.open("cat.jpeg").convert("RGB")
img = img.resize((224, 224))

img_np = np.array(img)

# ---------- FP32 ----------

fp32_input = (img_np / 255.0).astype(np.float32)
fp32_input = np.expand_dims(fp32_input, axis=0)

input_idx = fp32.get_input_details()[0]["index"]
output_idx = fp32.get_output_details()[0]["index"]

fp32.set_tensor(input_idx, fp32_input)

start = time.time()
fp32.invoke()
fp32_latency = (time.time() - start) * 1000

fp32_output = fp32.get_tensor(output_idx)

print("FP32 LATENCY:", round(fp32_latency, 2), "ms")
print("FP32 TOP CLASS:", np.argmax(fp32_output))

# ---------- INT8 ----------

input_details = int8.get_input_details()[0]

scale = input_details["quantization"][0]
zero_point = input_details["quantization"][1]

int8_input = (img_np / 255.0)
int8_input = int8_input / scale + zero_point
int8_input = int8_input.astype(np.int8)
int8_input = np.expand_dims(int8_input, axis=0)

input_idx = int8.get_input_details()[0]["index"]
output_idx = int8.get_output_details()[0]["index"]

int8.set_tensor(input_idx, int8_input)

start = time.time()
int8.invoke()
int8_latency = (time.time() - start) * 1000

int8_output = int8.get_tensor(output_idx)

print("INT8 LATENCY:", round(int8_latency, 2), "ms")
print("INT8 TOP CLASS:", np.argmax(int8_output))