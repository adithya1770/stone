import sys, os, csv, subprocess, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.inference_engine import InferenceEngine
from runtime.telemetry import get_telemetry
from runtime.decision_engine import get_algo, choose, update, get_last_source, set_extra_telemetry
from runtime.reward import calculate_reward

FOLDER = "datasets/custom_test"
LABELS_FILE = os.path.join(FOLDER, "labels.csv")
OUT_DIR = "results/custom_suite"
ITERATIONS = 60          # cycles through your 14 images ~4x each
STRESS_START_AT = 30     # stress-ng kicks in halfway through each config's run
STRESS_DURATION = 45

os.makedirs(OUT_DIR, exist_ok=True)

engine = InferenceEngine(
    fp32_path="models/mobilenet_v2_fp32.tflite",
    int8_path="models/mobilenet_v2_int8.tflite",
    labels_path="models/labels.txt"
)

true_labels = {}
with open(LABELS_FILE) as f:
    for row in csv.DictReader(f):
        true_labels[row["filename"]] = row["true_label"]

image_files = [f for f in os.listdir(FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
image_files.sort()

# cycle images to fill ITERATIONS
pool = [image_files[i % len(image_files)] for i in range(ITERATIONS)]


def run_experiment(name, model_override=None, algo=None):
    print(f"\n=== Running {name} ===")
    rows = []
    stress_proc = None

    for i, fname in enumerate(pool):
        if i == STRESS_START_AT:
            print(f"  [stress-ng starting for {STRESS_DURATION}s]")
            stress_proc = subprocess.Popen(
                ["stress-ng", "--cpu", "0", "--timeout", f"{STRESS_DURATION}s"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            time.sleep(1.5)  # let load ramp up

        path = os.path.join(FOLDER, fname)
        t = get_telemetry()

        if algo is not None:
            set_extra_telemetry(algo, t["cpu_freq_current"], t["cpu_freq_max"], t["disk_busy_time"])
            decision_start = time.time()
            chosen_model, scores = choose(algo, t["cpu"], t["ram"], t["temperature"], t["battery"])
            decision_time_ms = round((time.time() - decision_start) * 1000, 3)
            source = get_last_source(algo) or "algo"
        else:
            chosen_model = model_override
            decision_time_ms = 0.0
            source = "static"

        result = engine.run(path, chosen_model)

        reward = calculate_reward(
            confidence=result["confidence"], latency_ms=result["latency_ms"],
            cpu=t["cpu"], ram=t["ram"], temperature=t["temperature"],
            decision_time_ms=decision_time_ms
        )
        if algo is not None:
            update(algo, chosen_model, t["cpu"], t["ram"], t["temperature"], t["battery"],
                   reward, confidence=result["confidence"])

        true_label = true_labels.get(fname, "unknown")
        correct = "" if true_label == "unknown" else str(result["label"].strip().lower() == true_label.strip().lower())

        rows.append({
            "iteration": i, "file": fname, "cpu": t["cpu"], "ram": t["ram"],
            "temperature": t["temperature"], "true_label": true_label,
            "predicted": result["label"], "correct": correct,
            "model": result["model"], "decision_source": source,
            "confidence": result["confidence"], "latency_ms": result["latency_ms"],
            "decision_time_ms": decision_time_ms
        })
        print(f"  [{i}] {fname} -> {result['label']} ({chosen_model}, cpu={t['cpu']}%) correct={correct}")

    if stress_proc:
        stress_proc.wait()

    out_path = os.path.join(OUT_DIR, f"{name}_log.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved -> {out_path}")


# --- Run all configs, matching your standard 6-experiment suite ---
run_experiment("baseline", model_override="fp32")
run_experiment("always_int8", model_override="int8")
run_experiment("linucb_cold", algo=get_algo("linucb", alpha=1.0, state_path="logging/custom_linucb_cold.npz"))
run_experiment("linucb_warm", algo=get_algo("linucb", alpha=1.0, state_path="logging/linucb_state.npz"))
run_experiment("egreedy", algo=get_algo("egreedy", epsilon=0.1, state_path="logging/custom_egreedy.npz"))
run_experiment("eightsignal_cold", algo=get_algo("eightsignal", alpha=1.0, state_path="logging/custom_es_cold.npz"))
run_experiment("eightsignal_warm", algo=get_algo("eightsignal", alpha=1.0, state_path="logging/eight_signal_linucb_state.npz"))

print("\nAll custom-suite experiments complete. Results in results/custom_suite/")