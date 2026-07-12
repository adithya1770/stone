import numpy as np
from PIL import Image
import time
import ai_edge_litert.interpreter as tflite


class InferenceEngine:

    def __init__(self, fp32_path, int8_path, labels_path):
        self.fp32 = tflite.Interpreter(model_path=fp32_path)
        self.fp32.allocate_tensors()

        self.int8 = tflite.Interpreter(model_path=int8_path)
        self.int8.allocate_tensors()

        with open(labels_path) as f:
            self.labels = [line.strip() for line in f]

        self._warmup()
        print("Engine ready.")

    def _warmup(self):
        print("Running warm-up inference...")

        dummy_fp32 = np.zeros((1, 224, 224, 3), dtype=np.float32)
        self.fp32.set_tensor(
            self.fp32.get_input_details()[0]['index'], dummy_fp32)
        self.fp32.invoke()

        dummy_int8 = np.zeros((1, 224, 224, 3), dtype=np.int8)
        self.int8.set_tensor(
            self.int8.get_input_details()[0]['index'], dummy_int8)
        self.int8.invoke()

        print("Warm-up complete.")

    def run(self, image_path, model_type):
        model = self.fp32 if model_type == "fp32" else self.int8
        is_int8 = model_type == "int8"

        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = np.array(img)
        img_array = (img_array.astype(np.int32) - 128).astype(np.int8) \
            if is_int8 else (img_array / 255.0).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        inp = model.get_input_details()[0]['index']
        out = model.get_output_details()[0]['index']

        model.set_tensor(inp, img_array)
        start = time.time()
        model.invoke()
        latency_ms = round((time.time() - start) * 1000, 2)

        raw_output = model.get_tensor(out).flatten()

        if is_int8:
            output_details = model.get_output_details()[0]
            scale = output_details['quantization_parameters']['scales'][0]
            zero_point = output_details['quantization_parameters']['zero_points'][0]
            output = scale * (raw_output.astype(np.float32) - zero_point)
        else:
            output = raw_output.astype(np.float32)

        top_index = int(np.argmax(output))
        label = self.labels[top_index] if top_index < len(self.labels) else "unknown"

        return {
            "model":      model_type,
            "label":      label,
            "confidence": round(float(np.max(output)), 4),
            "latency_ms": latency_ms
        }