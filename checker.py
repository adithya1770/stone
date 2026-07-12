# run_fixed_comparison.py

import csv
import os
from model_manager import ModelManager

TEST_IMAGES_DIR = "test_images"
OUTPUT_CSV = "fixed_comparison.csv"

def load_ground_truth(csv_path):
    gt = {}
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            gt[row["filename"]] = row["ground_truth_label"]
    return gt

ground_truth = load_ground_truth(os.path.join(TEST_IMAGES_DIR, "ground_truth.csv"))
mm = ModelManager("models/mobilenet_v2_fp32.tflite", "models/mobilenet_v2_int8.tflite", "models/labels.txt")

rows = []
for filename, gt_label in sorted(ground_truth.items()):
    image_path = os.path.join(TEST_IMAGES_DIR, filename)
    fp32_result = mm.predict(image_path, "fp32")
    int8_result = mm.predict(image_path, "int8")
    rows.append({
        "filename": filename, "ground_truth": gt_label,
        "fp32_label": fp32_result["label"], "fp32_correct": int(fp32_result["label"] == gt_label),
        "fp32_confidence": fp32_result["confidence"], "fp32_latency_ms": fp32_result["latency_ms"],
        "int8_label": int8_result["label"], "int8_correct": int(int8_result["label"] == gt_label),
        "int8_confidence": int8_result["confidence"], "int8_latency_ms": int8_result["latency_ms"],
    })

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} paired comparisons to {OUTPUT_CSV}")