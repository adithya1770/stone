import tensorflow as tf
import numpy as np
from PIL import Image
import time


class ModelManager:

    def __init__(self, fp32_path, int8_path, labels_path):

        self.fp32 = tf.lite.Interpreter(model_path=fp32_path)
        self.fp32.allocate_tensors()

        self.int8 = tf.lite.Interpreter(model_path=int8_path)
        self.int8.allocate_tensors()

        with open(labels_path, "r") as f:
            self.labels = [line.strip() for line in f]

        print("ModelManager initialized.")

    def predict(self, image_path, model_type):

        model = self.fp32 if model_type == "fp32" else self.int8

        # Load image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))

        img_np = np.array(img)

        # FP32 preprocessing
        if model_type == "fp32":

            input_data = (img_np / 255.0).astype(np.float32)

        # INT8 preprocessing
        else:

            input_details = model.get_input_details()[0]

            scale = input_details["quantization"][0]
            zero_point = input_details["quantization"][1]

            input_data = (img_np / 255.0)
            input_data = input_data / scale + zero_point
            input_data = input_data.astype(np.int8)

        input_data = np.expand_dims(input_data, axis=0)

        # Get tensor indices
        input_index = model.get_input_details()[0]["index"]
        output_index = model.get_output_details()[0]["index"]

        # Set input
        model.set_tensor(input_index, input_data)

        # Run inference
        start = time.time()

        model.invoke()

        latency_ms = (time.time() - start) * 1000

        # Read output
        output = model.get_tensor(output_index)

        if model_type == "int8":
            output_details = model.get_output_details()[0]
            scale = output_details["quantization"][0]
            zero_point = output_details["quantization"][1]

            output = (output.astype(np.float32) - zero_point) * scale
        

        class_id = int(np.argmax(output))
        confidence = float(np.max(output))

        # Safety check
        if class_id < len(self.labels):
            label = self.labels[class_id]
        else:
            label = "unknown"

        return {
            "model": model_type,
            "class_id": class_id,
            "label": label,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2)
        }