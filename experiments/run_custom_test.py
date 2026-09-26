import sys, os, csv, subprocess, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.inference_engine import InferenceEngine
from runtime.telemetry import get_telemetry
from runtime.decision_engine import get_algo, choose, update, get_last_source, set_extra_telemetry
from runtime.reward import calculate_reward

FOLDER = "datasets/custom_test"
LABELS_FILE = os.path.join(FOLDER, "labels.csv")
OUT_FILE = "logging/custom_test_results.csv"
STRESS_DURATION = 90  # seconds — enough to cover a full pass over all algos/images

engine = InferenceEngine(
    fp32_path="models/mobilenet_v2_fp32.tflite",
    int8_path="models/mobilenet_v2_int8.tflite",
    labels_path="models/labels.txt"
)

true_labels = {}
with open(LABELS_FILE) as f:
    for row in csv.DictReader(f):
        true_labels[row["filename"]] = row["true_label"]

algos_to_test = {
    "linucb": get_algo("linucb", alpha=1.0, state_path="logging/custom_linucb.npz"),
    "egreedy": get_algo("egreedy", epsilon=0.1, state_path="logging/custom_egreedy.npz"),
    "eightsignal": get_algo("eightsignal", alpha=1.0, state_path="logging/custom_eightsignal.npz"),
}

results = []
image_files = [f for f in os.listdir(FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"Starting stress-ng for {STRESS_DURATION}s in background so algorithms see real variation...")
stress_proc = subprocess.Popen(["stress-ng", "--cpu", "0", "--timeout", f"{STRESS_DURATION}s"])
time.sleep(2)  # let stress ramp up before first reading

for algo_name, algo in algos_to_test.items():
    for fname in image_files:
        path = os.path.join(FOLDER, fname)
        t = get_telemetry()
        set_extra_telemetry(algo, t["cpu_freq_current"], t["cpu_freq_max"], t["disk_busy_time"])

        chosen_model, scores = choose(algo, t["cpu"], t["ram"], t["temperature"], t["battery"])
        source = get_last_source(algo) or "algo"
        result = engine.run(path, chosen_model)

        reward = calculate_reward(
            confidence=result["confidence"], latency_ms=result["latency_ms"],
            cpu=t["cpu"], ram=t["ram"], temperature=t["temperature"], decision_time_ms=0.3
        )
        update(algo, chosen_model, t["cpu"], t["ram"], t["temperature"], t["battery"],
               reward, confidence=result["confidence"])

        true_label = true_labels.get(fname, "unknown")
        correct = "" if true_label == "unknown" else str(result["label"].strip().lower() == true_label.strip().lower())

        row = {
            "algo": algo_name, "file": fname, "true_label": true_label,
            "predicted": result["label"], "correct": correct,
            "model": result["model"], "decision_source": source,
            "cpu": t["cpu"], "temperature": t["temperature"],
            "confidence": result["confidence"], "latency_ms": result["latency_ms"]
        }
        results.append(row)
        print(f"[{algo_name}] {fname} -> {result['label']} ({chosen_model}, cpu={t['cpu']}%) correct={correct}")

stress_proc.wait()

with open(OUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone. Results saved to {OUT_FILE}")