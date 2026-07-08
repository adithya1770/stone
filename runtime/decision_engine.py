import numpy as np

class LinUCB:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.models = ["fp32", "int8"]
        self.d = 3

        self.A = {m: np.identity(self.d) for m in self.models}
        self.b = {m: np.zeros(self.d) for m in self.models}

    def _context(self, cpu, ram, temp):
        return np.array([cpu / 100.0, ram / 100.0, temp / 100.0])

    def choose(self, cpu, ram, temp):
        x = self._context(cpu, ram, temp)
        scores = {}

        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            expected = theta @ x
            uncertainty = self.alpha * np.sqrt(x @ A_inv @ x)
            scores[model] = expected + uncertainty

        chosen = max(scores, key=scores.get)
        return chosen, scores

    def update(self, model, cpu, ram, temp, reward):
        x = self._context(cpu, ram, temp)
        self.A[model] += np.outer(x, x)
        self.b[model] += reward * x

HIGH_THRESHOLD = 70
LOW_THRESHOLD = 50
REQUIRED_READINGS = 10


def health_score(cpu, ram, temp):
    return (0.5 * cpu) + (0.2 * ram) + (0.3 * temp)


def decide(cpu, ram, temp, state):
    health = health_score(cpu, ram, temp)

    if health > HIGH_THRESHOLD:
        state["high_count"] += 1
        state["low_count"] = 0
    elif health < LOW_THRESHOLD:
        state["low_count"] += 1
        state["high_count"] = 0
    else:
        state["high_count"] = 0
        state["low_count"] = 0

    if state["high_count"] >= REQUIRED_READINGS:
        state["current_model"] = "int8"
        state["high_count"] = 0

    if state["low_count"] >= REQUIRED_READINGS:
        state["current_model"] = "fp32"
        state["low_count"] = 0

    return {
        "model": state["current_model"],
        "health_score": round(health, 2)
    }