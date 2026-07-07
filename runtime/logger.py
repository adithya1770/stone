import csv
import os

LOG_FILE = "logging/session_log.csv"


def initialize_logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "cpu",
                "ram",
                "temperature",
                "health_score",
                "model",
                "label",
                "confidence",
                "latency_ms"
            ])


def log_data(data):
    with open(LOG_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            data["timestamp"],
            data["cpu"],
            data["ram"],
            data["temperature"],
            data["health_score"],
            data["model"],
            data["label"],
            data["confidence"],
            data["latency_ms"]
        ])