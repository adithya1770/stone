import sys
import os
import time
import csv
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from runtime.inference_engine import InferenceEngine
from runtime.telemetry import get_telemetry
from runtime.decision_engine import get_algo, choose, update, get_last_source, set_extra_telemetry
from runtime.reward import calculate_reward
from runtime.logger import initialize_logger

app = FastAPI()

engine = InferenceEngine(
    fp32_path="models/mobilenet_v2_fp32.tflite",
    int8_path="models/mobilenet_v2_int8.tflite",
    labels_path="models/labels.txt"
)

algo = get_algo("eightsignal", alpha=1.0)

LOG_FILE = "logging/web_demo_log.csv"

FIELDS = [
    "request_id", "timestamp", "cpu", "ram", "temperature", "battery",
    "health_score", "model", "decision_source", "true_label", "label",
    "correct", "confidence", "latency_ms", "decision_time_ms"
]


def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDS)


def append_log(row_dict):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(row_dict)


def update_log_label(request_id, true_label):
    """Rewrites the CSV to fill in the true_label + correct fields for one
    row, matched by request_id. Fine for a live demo's log volume."""
    if not os.path.exists(LOG_FILE):
        return
    with open(LOG_FILE, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row["request_id"] == str(request_id):
            row["true_label"] = true_label
            row["correct"] = str(row["label"].strip().lower() == true_label.strip().lower())

    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


init_log()

app.mount("/static", StaticFiles(directory="web_demo/static"), name="static")

_request_counter = {"n": 0}


@app.get("/")
def home():
    return FileResponse("web_demo/static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "device": "raspberry-pi"}


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    temp_path = "web_demo/last_frame.jpg"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    t = get_telemetry()
    set_extra_telemetry(algo, t["cpu_freq_current"], t["cpu_freq_max"], t["disk_busy_time"])

    decision_start = time.time()
    chosen_model, scores = choose(algo, t["cpu"], t["ram"], t["temperature"], t["battery"])
    decision_time_ms = round((time.time() - decision_start) * 1000, 3)
    source = get_last_source(algo) or "algo"

    result = engine.run(temp_path, chosen_model)

    reward = calculate_reward(
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        cpu=t["cpu"], ram=t["ram"], temperature=t["temperature"],
        decision_time_ms=decision_time_ms
    )
    update(algo, chosen_model, t["cpu"], t["ram"], t["temperature"], t["battery"],
           reward, confidence=result["confidence"])

    _request_counter["n"] += 1
    request_id = _request_counter["n"]

    append_log({
        "request_id": request_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": t["cpu"], "ram": t["ram"], "temperature": t["temperature"],
        "battery": t["battery"],
        "health_score": round(scores.get("fp32", 0), 4),
        "model": result["model"],
        "decision_source": source,
        "true_label": "",
        "label": result["label"],
        "correct": "",
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
        "decision_time_ms": decision_time_ms
    })

    return {
        "request_id": request_id,
        "model": result["model"],
        "label": result["label"],
        "confidence": result["confidence"],
        "latency_ms": result["latency_ms"],
        "decision_time_ms": decision_time_ms,
        "decision_source": source,
        "cpu": t["cpu"], "ram": t["ram"], "temperature": t["temperature"]
    }


class LabelPayload(BaseModel):
    request_id: int
    true_label: str


@app.post("/label")
def label(payload: LabelPayload):
    update_log_label(payload.request_id, payload.true_label)
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)