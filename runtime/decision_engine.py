import numpy as np
import os
from collections import deque
from runtime.telemetry import get_telemetry


class RollingBaseline:
    def __init__(self, window=10):
        self.history = deque(maxlen=window)

    def update(self, value):
        self.history.append(value)

    def mean(self):
        return sum(self.history) / len(self.history) if self.history else 0.0

    def std(self):
        if len(self.history) < 2:
            return 0.0
        m = self.mean()
        var = sum((x - m) ** 2 for x in self.history) / len(self.history)
        return var ** 0.5

    def z(self, value):
        if not self.history:
            return 0.0
        return (value - self.mean()) / (self.std() + 1e-6)

    def ready(self):
        return len(self.history) >= 3


class LinUCB:
    def __init__(self, alpha=1.0, state_path="logging/linucb_state.npz"):
        self.alpha = alpha
        self.models = ["fp32", "int8"]
        self.d = 5
        self.state_path = state_path

        if os.path.exists(state_path):
            self._load()
            print("LinUCB: loaded previous state.")
        else:
            self.A = {m: np.identity(self.d) for m in self.models}
            self.b = {m: np.zeros(self.d) for m in self.models}
            print("LinUCB: starting fresh.")

    def _context(self, cpu, ram, temp, battery):
        return np.array([cpu / 100.0, ram / 100.0, temp / 100.0, battery / 100.0, 1.0])

    def choose(self, cpu, ram, temp, battery):
        x = self._context(cpu, ram, temp, battery)
        scores = {}
        for model in self.models:
            A_inv = np.linalg.inv(self.A[model])
            theta = A_inv @ self.b[model]
            expected = theta @ x
            uncertainty = self.alpha * np.sqrt(x @ A_inv @ x)
            scores[model] = expected + uncertainty
        chosen = max(scores, key=scores.get)
        return chosen, scores

    def update(self, model, cpu, ram, temp, battery, reward, confidence=None):
        x = self._context(cpu, ram, temp, battery)
        self.A[model] += np.outer(x, x)
        self.b[model] += reward * x
        self._save()

    def _save(self):
        np.savez(self.state_path,
                  A_fp32=self.A["fp32"], A_int8=self.A["int8"],
                  b_fp32=self.b["fp32"], b_int8=self.b["int8"])

    def _load(self):
        data = np.load(self.state_path)
        self.A = {"fp32": data["A_fp32"], "int8": data["A_int8"]}
        self.b = {"fp32": data["b_fp32"], "int8": data["b_int8"]}


class EpsilonGreedy:
    def __init__(self, epsilon=0.1, state_path="logging/egreedy_state.npz"):
        self.epsilon = epsilon
        self.models = ["fp32", "int8"]
        self.state_path = state_path

        if os.path.exists(state_path):
            data = np.load(state_path)
            self.counts = {m: float(data[f"count_{m}"]) for m in self.models}
            self.totals = {m: float(data[f"total_{m}"]) for m in self.models}
            print("EpsilonGreedy: loaded previous state.")
        else:
            self.counts = {m: 0.0 for m in self.models}
            self.totals = {m: 0.0 for m in self.models}
            print("EpsilonGreedy: starting fresh.")

    def _avg_reward(self, model):
        if self.counts[model] == 0:
            return 0.0
        return self.totals[model] / self.counts[model]

    def choose(self, cpu, ram, temp, battery):
        if np.random.random() < self.epsilon:
            chosen = np.random.choice(self.models)
        else:
            chosen = max(self.models, key=self._avg_reward)
        scores = {m: round(self._avg_reward(m), 4) for m in self.models}
        return chosen, scores

    def update(self, model, cpu, ram, temp, battery, reward, confidence=None):
        self.counts[model] += 1
        self.totals[model] += reward
        np.savez(self.state_path,
                  **{f"count_{m}": self.counts[m] for m in self.models},
                  **{f"total_{m}": self.totals[m] for m in self.models})


class EightSignalController:
    THERMAL_LIMIT = 90.0
    TAU = 0.25
    MIN_DWELL = 3  # minimum iterations on int8 before switching back to fp32

    # Absolute overrides: these catch sustained stress that the adaptive
    # RollingBaseline would otherwise "normalize away" after ~10 iterations
    # of consistently high readings (the baseline's own mean/std drifts to
    # match the stressed state, so z-scores decay toward 0 even though the
    # device is still genuinely stressed). These force a strong int8 vote
    # regardless of what the rolling baseline currently considers "normal".
    CPU_ABSOLUTE_THRESHOLD = 85.0       # % utilization
    THERMAL_ABSOLUTE_THRESHOLD = 10.0   # headroom in °C (i.e. temp >= 80.0)

    def get_last_score(self):
        return self.last_S

    def __init__(self, alpha=1.0, state_path="logging/eight_signal_linucb_state.npz"):
        self.linucb = LinUCB(alpha=alpha, state_path=state_path)

        self.baselines = {
            "cpu_util":         RollingBaseline(),
            "cpu_freq_ratio":   RollingBaseline(),
            "cpu_trend":        RollingBaseline(),
            "mem_pressure":     RollingBaseline(),
            "mem_activity":     RollingBaseline(),
            "thermal_headroom": RollingBaseline(),
            "power_rate":       RollingBaseline(),
            "difficulty":       RollingBaseline(),
        }

        self.prev_cpu = None
        self.prev_battery = None
        self.prev_disk_busy = None
        self.prev_confidence = 0.5
        self.last_source = None
        self.last_votes = None
        self.last_S = 0.0

        self._cpu_freq_current = 0.0
        self._cpu_freq_max = 4000.0
        self._disk_busy = 0

        # hysteresis state
        self.current_model = None
        self.iterations_on_current = 0

    def set_extra_telemetry(self, cpu_freq_current, cpu_freq_max, disk_busy_time):
        self._cpu_freq_current = cpu_freq_current
        self._cpu_freq_max = cpu_freq_max
        self._disk_busy = disk_busy_time

    def _vote(self, name, current_value, higher_is_worse):
        """Continuous vote in [-1, 1] based on z-score magnitude, instead of a
        hard -1/0/+1 snap. A mildly-off signal now contributes less than a
        wildly-off one. Capped at |z|=2.0 so outliers don't dominate."""
        baseline = self.baselines[name]
        baseline.update(current_value)
        if not baseline.ready():
            return 0.0
        z = baseline.z(current_value)
        scaled = max(-1.0, min(1.0, z / 2.0))
        if higher_is_worse:
            return -scaled
        else:
            return scaled

    def choose(self, cpu, ram, temp, battery):
        v1 = self._vote("cpu_util", cpu, higher_is_worse=True)
        # Absolute override: sustained high CPU keeps voting toward int8 even
        # after the rolling baseline has adapted to treat it as "normal".
        if cpu >= self.CPU_ABSOLUTE_THRESHOLD:
            v1 = min(v1, -1.0)

        freq_ratio = self._cpu_freq_current / (self._cpu_freq_max or 1.0)
        v2 = self._vote("cpu_freq_ratio", freq_ratio, higher_is_worse=False)

        trend = 0.0 if self.prev_cpu is None else (cpu - self.prev_cpu)
        self.prev_cpu = cpu
        v3 = self._vote("cpu_trend", trend, higher_is_worse=True)

        mem_pressure = ram / 100.0
        v4 = self._vote("mem_pressure", mem_pressure, higher_is_worse=True)

        activity = 0.0 if self.prev_disk_busy is None else max(0, self._disk_busy - self.prev_disk_busy)
        self.prev_disk_busy = self._disk_busy
        v5 = self._vote("mem_activity", activity, higher_is_worse=True)

        headroom = self.THERMAL_LIMIT - temp
        v6 = self._vote("thermal_headroom", headroom, higher_is_worse=False)
        # Absolute override: low thermal headroom keeps voting toward int8
        # even after the baseline has adapted to sustained heat.
        if headroom <= self.THERMAL_ABSOLUTE_THRESHOLD:
            v6 = min(v6, -1.0)

        rate = 0.0 if self.prev_battery is None else max(0.0, self.prev_battery - battery)
        self.prev_battery = battery
        v7 = self._vote("power_rate", rate, higher_is_worse=True)

        difficulty = 1.0 - self.prev_confidence
        v8 = self._vote("difficulty", difficulty, higher_is_worse=False)

        votes = [v1, v2, v3, v4, v5, v6, v7, v8]
        S = sum(votes) / len(votes)
        self.last_votes = votes
        self.last_S = S

        if S > self.TAU:
            desired_model, scores, source = "fp32", {"fp32": S, "int8": -S}, "vote_fp32"
        elif S < -self.TAU:
            desired_model, scores, source = "int8", {"fp32": S, "int8": -S}, "vote_int8"
        else:
            desired_model, scores = self.linucb.choose(cpu, ram, temp, battery)
            source = "linucb"

        final_model = self._apply_hysteresis(desired_model)
        self.last_source = source

        return final_model, scores

    def _apply_hysteresis(self, desired_model):
        """Asymmetric dwell: switching INTO int8 is always immediate.
        Switching back OUT of int8 to fp32 requires MIN_DWELL iterations of
        stability first, so a brief dip in stress doesn't immediately bounce
        it back to the slower model."""
        if self.current_model is None:
            self.current_model = desired_model
            self.iterations_on_current = 1
            return desired_model

        if desired_model == self.current_model:
            self.iterations_on_current += 1
            return self.current_model

        if desired_model == "int8":
            self.current_model = "int8"
            self.iterations_on_current = 1
            return "int8"

        if self.iterations_on_current >= self.MIN_DWELL:
            self.current_model = "fp32"
            self.iterations_on_current = 1
            return "fp32"
        else:
            self.iterations_on_current += 1
            return self.current_model

    def update(self, model, cpu, ram, temp, battery, reward, confidence=None):
        if confidence is not None:
            self.prev_confidence = confidence
        if self.last_source == "linucb":
            self.linucb.update(model, cpu, ram, temp, battery, reward)

    def get_last_source(self):
        return self.last_source


def get_algo(name="linucb", **kwargs):
    name = name.lower().strip()
    if name == "linucb":
        return LinUCB(**kwargs)
    elif name in ("egreedy", "epsilon"):
        kwargs.pop("alpha", None)
        return EpsilonGreedy(**kwargs)
    elif name in ("eightsignal", "combo", "final"):
        return EightSignalController(**kwargs)
    else:
        raise ValueError(f"Unknown algorithm '{name}'. Choose 'linucb', 'egreedy', or 'eightsignal'.")