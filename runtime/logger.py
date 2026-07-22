import csv
import os

DEFAULT_LOG = "logging/session_log.csv"


def initialize_logger(log_file=DEFAULT_LOG):
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "cpu",
                "ram",
                "temperature",
                "health_score",
                "model",
                "label",
                "confidence",
                "latency_ms",
                "decision_time_ms"
            ])
    return log_file


def log_data(data, log_file=DEFAULT_LOG):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            data["timestamp"],
            data["cpu"],
            data["ram"],
            data["temperature"],
            data["health_score"],
            data["model"],
            data["label"],
            data["confidence"],
            data["latency_ms"],
            data.get("decision_time_ms", 0)
        ])