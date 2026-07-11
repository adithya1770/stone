import os
from model_manager import ModelManager

VAL_DIR = "imagenette2-160/val"

IMAGENETTE_LABELS = {
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

manager = ModelManager(
    "models/mobilenet_v2_fp32.tflite",
    "models/mobilenet_v2_int8.tflite",
    "models/labels.txt"
)


def evaluate(model_type, max_images=500):

    total_latency = 0
    total = 0
    correct = 0

    print(f"\nEvaluating {model_type.upper()}")

    for class_folder in sorted(os.listdir(VAL_DIR)):

        class_path = os.path.join(VAL_DIR, class_folder)

        if not os.path.isdir(class_path):
            continue

        ground_truth = IMAGENETTE_LABELS[class_folder]

        for image_name in os.listdir(class_path):

            image_path = os.path.join(class_path, image_name)

            try:

                result = manager.predict(
                    image_path,
                    model_type
                )

                total += 1
                total_latency += result["latency_ms"]

                if result["label"] == ground_truth:
                    correct += 1

                if total % 50 == 0:
                    print(f"Processed {total} images...")

                if total >= max_images:
                    break

            except Exception as e:

                print(
                    f"Failed on {image_path}: {e}"
                )

        if total >= max_images:
            break

    accuracy = (correct / total) * 100
    avg_latency = total_latency / total

    print("\n========== RESULTS ==========")
    print(f"Model: {model_type.upper()}")
    print(f"Images Tested : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.2f}%")
    print(f"Avg Latency   : {avg_latency:.2f} ms")
    print("=============================\n")


evaluate("fp32", max_images=1000)
evaluate("int8", max_images=1000)