import os
import sys
import csv
import random
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.inference_engine import InferenceEngine

DATASET_DIR = "datasets/imagenette2-320"
VAL_DIR = os.path.join(DATASET_DIR, "val")
LABELS_FILE = "models/labels.txt"

# Imagenette's 10 WordNet IDs mapped to their real, full ImageNet names.
# These are looked up by hand from the standard ImageNet class list —
# this is a small, fixed list, so no need for a words.txt lookup step.
WNID_TO_NAME = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=100,
                     help="Number of validation images to test (sampled across all 10 classes)")
parser.add_argument("--seed", type=int, default=42, help="Random sample seed, for repeatability")
args = parser.parse_args()


def load_labels(path):
    with open(path) as f:
        return [line.strip() for line in f]


def build_wnid_to_label_index(labels):
    label_lookup = {name.strip().lower(): idx for idx, name in enumerate(labels)}
    wnid_to_index = {}
    unmatched = []

    for wnid, name in WNID_TO_NAME.items():
        key = name.strip().lower()
        if key in label_lookup:
            wnid_to_index[wnid] = label_lookup[key]
        else:
            unmatched.append((wnid, name))

    return wnid_to_index, unmatched


print("Loading label files...")
labels = load_labels(LABELS_FILE)
wnid_to_label_index, unmatched = build_wnid_to_label_index(labels)

print(f"Matched {len(wnid_to_label_index)} / {len(WNID_TO_NAME)} Imagenette classes to labels.txt entries.")
if unmatched:
    print("Unmatched classes (check spelling against labels.txt manually):")
    for wnid, name in unmatched:
        print(f"  {wnid}: '{name}'")

image_entries = []
for wnid in wnid_to_label_index:
    class_dir = os.path.join(VAL_DIR, wnid)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        if fname.lower().endswith((".jpeg", ".jpg", ".png")):
            image_entries.append((os.path.join(class_dir, fname), wnid))

print(f"\nTotal usable validation images: {len(image_entries)}")

random.seed(args.seed)
sample_size = min(args.samples, len(image_entries))
sampled = random.sample(image_entries, sample_size)

print(f"Testing on {sample_size} sampled images.\n")

engine = InferenceEngine(
    fp32_path="models/mobilenet_v2_fp32.tflite",
    int8_path="models/mobilenet_v2_int8.tflite",
    labels_path=LABELS_FILE
)

fp32_correct = 0
int8_correct = 0
results_rows = []

for i, (image_path, wnid) in enumerate(sampled):
    true_index = wnid_to_label_index[wnid]
    true_label = labels[true_index]

    fp32_result = engine.run(image_path, "fp32")
    int8_result = engine.run(image_path, "int8")

    fp32_hit = fp32_result["label"].strip().lower() == true_label.strip().lower()
    int8_hit = int8_result["label"].strip().lower() == true_label.strip().lower()

    fp32_correct += int(fp32_hit)
    int8_correct += int(int8_hit)

    results_rows.append({
        "filename": os.path.basename(image_path),
        "true_label": true_label,
        "fp32_predicted": fp32_result["label"],
        "fp32_correct": fp32_hit,
        "fp32_confidence": fp32_result["confidence"],
        "int8_predicted": int8_result["label"],
        "int8_correct": int8_hit,
        "int8_confidence": int8_result["confidence"],
    })

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{sample_size} done — "
              f"fp32 running acc: {round(100*fp32_correct/(i+1), 1)}%, "
              f"int8 running acc: {round(100*int8_correct/(i+1), 1)}%")

fp32_acc = round(100 * fp32_correct / sample_size, 2)
int8_acc = round(100 * int8_correct / sample_size, 2)

os.makedirs("logging", exist_ok=True)
out_path = "logging/accuracy_test_results.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
    writer.writeheader()
    writer.writerows(results_rows)

print("\n" + "=" * 50)
print("ACCURACY TEST RESULTS")
print("=" * 50)
print(f"Samples tested   : {sample_size}")
print(f"FP32 accuracy    : {fp32_acc}% ({fp32_correct}/{sample_size})")
print(f"INT8 accuracy    : {int8_acc}% ({int8_correct}/{sample_size})")
print(f"Accuracy gap     : {round(fp32_acc - int8_acc, 2)} percentage points")
print(f"\nDetailed per-image results saved to: {out_path}")