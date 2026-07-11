# prepare_test_set.py

import os
import shutil
import csv

IMAGENETTE_VAL_DIR = "imagenette2-160/val"
OUTPUT_DIR = "test_images"
IMAGES_PER_CLASS = 5

# Maps each Imagenette folder (WordNet synset ID) to its real class name —
# these names match your labels.txt exactly since both use standard ImageNet naming.
SYNSET_TO_LABEL = {
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

os.makedirs(OUTPUT_DIR, exist_ok=True)
rows = []

for synset, label in SYNSET_TO_LABEL.items():
    class_dir = os.path.join(IMAGENETTE_VAL_DIR, synset)
    images = sorted(os.listdir(class_dir))[:IMAGES_PER_CLASS]
    for img_name in images:
        dst_name = f"{synset}_{img_name}"
        shutil.copy(os.path.join(class_dir, img_name), os.path.join(OUTPUT_DIR, dst_name))
        rows.append([dst_name, label])

with open(os.path.join(OUTPUT_DIR, "ground_truth.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "ground_truth_label"])
    writer.writerows(rows)

print(f"Copied {len(rows)} images into {OUTPUT_DIR}/ with ground_truth.csv")