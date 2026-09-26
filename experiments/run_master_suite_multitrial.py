import sys, os, csv, subprocess, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.inference_engine import InferenceEngine
from runtime.telemetry import get_telemetry
from runtime.decision_engine import get_algo, choose, update, get_last_source, set_extra_telemetry
from runtime.reward import calculate_reward
from runtime.image_pool import ImagePool
from runtime.label_lookup import true_label_for_image, build_wnid_to_label_index

STRESS_START_AT = 40
STRESS_DURATION = 60
IMAGENETTE_ITERATIONS = 100
CUSTOM_ITERATIONS = 60
N_TRIALS = 3

engine = InferenceEngine(
    fp32_path="models/mobilenet_v2_fp32.tflite",
    int8_path="models/mobilenet_v2_int8.tflite",
    labels_path="models/labels.txt"
)

CONFIGS = [
    ("baseline",         None,               "fp32"),
    ("always_int8",      None,               "int8"),
    ("linucb_cold",      "linucb",           None),
    ("linucb_warm",      "linucb_warm",      None),
    ("egreedy",          "egreedy",          None),
    ("eightsignal_cold", "eightsignal",      None),
    ("eightsignal_warm", "eightsignal_warm", None),
]


def make_algo(kind, state_dir):
    if kind == "linucb":
        return get_algo("linucb", alpha=1.0, state_path=f"{state_dir}/linucb_cold_state.npz")
    if kind == "linucb_warm":
        return get_algo("linucb", alpha=1.0, state_path=f"{state_dir}/linucb_warm_state.npz")
    if kind == "egreedy":
        return get_algo("egreedy", epsilon=0.1, state_path=f"{state_dir}/egreedy_state.npz")
    if kind == "eightsignal":
        return get_algo("eightsignal", alpha=1.0, state_path=f"{state_dir}/eightsignal_cold_state.npz")
    if kind == "eightsignal_warm":
        return get_algo("eightsignal", alpha=1.0, state_path=f"{state_dir}/eightsignal_warm_state.npz")
    return None


def get_image_source(dataset):
    if dataset == "imagenette":
        pool = ImagePool(seed=42)
        n = min(IMAGENETTE_ITERATIONS, len(pool))
        wnid_to_index, labels = build_wnid_to_label_index("models/labels.txt")

        def get_item(i):
            path = pool.get(i)
            true_label = true_label_for_image(path, wnid_to_index, labels)
            return path, true_label

        return n, get_item
    else:
        folder = "datasets/custom_test"
        labels_file = os.path.join(folder, "labels.csv")
        true_labels = {}
        with open(labels_file) as f:
            for row in csv.DictReader(f):
                true_labels[row["filename"]] = row["true_label"]
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        n = CUSTOM_ITERATIONS
        pool_list = [files[i % len(files)] for i in range(n)]

        def get_item(i):
            fname = pool_list[i]
            return os.path.join(folder, fname), true_labels.get(fname, "unknown")

        return n, get_item


def run_config(name, algo_kind, model_override, dataset, out_dir, state_dir):
    print(f"  === [{dataset}] {name} ===")
    n_iterations, get_item = get_image_source(dataset)
    algo = make_algo(algo_kind, state_dir) if algo_kind else None

    rows = []
    stress_proc = None

    for i in range(n_iterations):
        if i == STRESS_START_AT:
            stress_proc = subprocess.Popen(
                ["stress-ng", "--cpu", "0", "--timeout", f"{STRESS_DURATION}s"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            time.sleep(1.5)

        path, true_label = get_item(i)
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

        true_label_str = true_label if true_label else "unknown"
        correct = "" if true_label_str == "unknown" else str(
            result["label"].strip().lower() == true_label_str.strip().lower()
        )

        rows.append({
            "iteration": i, "cpu": t["cpu"], "ram": t["ram"], "temperature": t["temperature"],
            "true_label": true_label_str, "predicted": result["label"], "correct": correct,
            "model": result["model"], "decision_source": source,
            "confidence": result["confidence"], "latency_ms": result["latency_ms"],
            "decision_time_ms": decision_time_ms
        })

    if stress_proc:
        stress_proc.wait()

    out_path = os.path.join(out_dir, f"{name}_log.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"    saved -> {out_path}")


def run_trial(trial_num, dataset):
    out_dir = f"results/pi_multitrial/trial{trial_num}/{dataset}"
    state_dir = f"logging/pi_multitrial_state/trial{trial_num}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    for name, algo_kind, model_override in CONFIGS:
        run_config(name, algo_kind, model_override, dataset, out_dir, state_dir)


if __name__ == "__main__":
    for trial in range(1, N_TRIALS + 1):
        print(f"\n{'='*60}\nTRIAL {trial}/{N_TRIALS}\n{'='*60}")
        run_trial(trial, "imagenette")
        run_trial(trial, "custom")

    print("\nAll trials complete. Results in results/pi_multitrial/trial1|2|3/")