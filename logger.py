import csv
import os


class RuntimeLogger:

    def __init__(self, log_file="logs/runtime_logs.csv"):

        self.log_file = log_file

        os.makedirs("logs", exist_ok=True)

        if not os.path.exists(self.log_file):

            with open(self.log_file, "w", newline="") as file:

                writer = csv.writer(file)

                writer.writerow([
                    "timestamp",
                    "cpu",
                    "cpu_ema",
                    "memory",
                    "temperature",
                    "chosen_model",
                    "latency",
                    "confidence",
                    "reward"
                ])

    def log(self, data):

        with open(self.log_file, "a", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                data["timestamp"],
                data["cpu"],
                data["cpu_ema"],
                data["memory"],
                data["temperature"],
                data["chosen_model"],
                data["latency"],
                data["confidence"],
                data["reward"]
            ])
from datetime import datetime


if __name__ == "__main__":

    logger = RuntimeLogger()

    logger.log({
        "timestamp": datetime.now(),
        "cpu": 20,
        "cpu_ema": 18,
        "memory": 30,
        "temperature": 45,
        "chosen_model": "FP32",
        "latency": 0.2,
        "confidence": 0.95,
        "reward": 0.90
    })

    print("Log written.")