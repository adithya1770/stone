

import csv
import os
import time

class Logger:
    """
    Writes one CSV row per inference event. Captures telemetry both
    before the decision was made and after inference completed, so you
    can later check whether running the model itself moved CPU/mem/temp.
    """

    FIELDNAMES = [
        "timestamp", "input_id",
        "cpu_raw_before", "cpu_ema_before", "mem_raw_before", "mem_ema_before", "temp_before",
        "cpu_raw_after", "cpu_ema_after", "mem_raw_after", "mem_ema_after", "temp_after",
        "chosen_action", "model_used",
        "predicted_class", "label", "confidence", "latency_ms",
        "ground_truth", "reward",  # left blank for now — filled in once reward/LinUCB is wired up
    ]

    def __init__(self, log_path: str):
        self.log_path = log_path
        file_is_new = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
        self._file = open(log_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if file_is_new:
            self._writer.writeheader()
            self._file.flush()

    def log(self, before_snap: dict, after_snap: dict, input_id, chosen_action, model_used,
             inference_result: dict, ground_truth=None, reward=None):
        row = {
            "timestamp": time.time(),
            "input_id": input_id,
            "cpu_raw_before": before_snap["cpu_raw"], "cpu_ema_before": before_snap["cpu_ema"],
            "mem_raw_before": before_snap["mem_raw"], "mem_ema_before": before_snap["mem_ema"],
            "temp_before": before_snap["temp"],
            "cpu_raw_after": after_snap["cpu_raw"], "cpu_ema_after": after_snap["cpu_ema"],
            "mem_raw_after": after_snap["mem_raw"], "mem_ema_after": after_snap["mem_ema"],
            "temp_after": after_snap["temp"],
            "chosen_action": chosen_action, "model_used": model_used,
            "predicted_class": inference_result["class_id"], "label": inference_result["label"],
            "confidence": inference_result["confidence"], "latency_ms": inference_result["latency_ms"],
            "ground_truth": ground_truth, "reward": reward,
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()