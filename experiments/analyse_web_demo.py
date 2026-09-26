import csv

LOG_FILE = "logging/web_demo_log.csv"


def load_labeled_rows():
    rows = []
    with open(LOG_FILE) as f:
        for row in csv.DictReader(f):
            if row["correct"] in ("True", "False"):
                rows.append(row)
    return rows


def accuracy(rows):
    if not rows:
        return None
    hits = sum(1 for r in rows if r["correct"] == "True")
    return round(100 * hits / len(rows), 2)


def precision_recall(rows, target_label):
    """Precision/recall for one specific class label, since multi-class
    precision/recall needs a per-class breakdown, not one global number."""
    predicted_positive = [r for r in rows if r["label"].strip().lower() == target_label.lower()]
    actual_positive = [r for r in rows if r["true_label"].strip().lower() == target_label.lower()]

    tp = sum(1 for r in predicted_positive if r["correct"] == "True")
    precision = round(100 * tp / len(predicted_positive), 2) if predicted_positive else None
    recall = round(100 * tp / len(actual_positive), 2) if actual_positive else None
    return precision, recall


rows = load_labeled_rows()
print(f"Total labeled frames: {len(rows)}")

if not rows:
    print("No labeled frames yet — use the pause button during the demo to label some.")
    exit()

print(f"\nOverall accuracy: {accuracy(rows)}%")

fp32_rows = [r for r in rows if r["model"] == "fp32"]
int8_rows = [r for r in rows if r["model"] == "int8"]
print(f"FP32 accuracy ({len(fp32_rows)} frames): {accuracy(fp32_rows)}%")
print(f"INT8 accuracy ({len(int8_rows)} frames): {accuracy(int8_rows)}%")

correct_rows = [r for r in rows if r["correct"] == "True"]
incorrect_rows = [r for r in rows if r["correct"] == "False"]
avg_conf_correct = round(sum(float(r["confidence"]) for r in correct_rows) / len(correct_rows), 3) if correct_rows else None
avg_conf_incorrect = round(sum(float(r["confidence"]) for r in incorrect_rows) / len(incorrect_rows), 3) if incorrect_rows else None
print(f"\nAvg confidence when correct: {avg_conf_correct}")
print(f"Avg confidence when incorrect: {avg_conf_incorrect}")

labels_seen = set(r["true_label"].strip().lower() for r in rows if r["true_label"])
print("\nPer-class precision/recall:")
for lbl in sorted(labels_seen):
    p, r = precision_recall(rows, lbl)
    print(f"  {lbl:<20} precision={p}  recall={r}")