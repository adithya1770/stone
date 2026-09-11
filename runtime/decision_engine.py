import numpy as np
import os
from collections import deque


# ---------- Rolling Baseline ----------

def rolling_baseline_new(window=10):
    return {"history": deque(maxlen=window)}


def rolling_baseline_update(baseline, value):
    baseline["history"].append(value)


def rolling_baseline_mean(baseline):
    history = baseline["history"]
    return sum(history) / len(history) if history else 0.0


def rolling_baseline_std(baseline):
    history = baseline["history"]
    if len(history) < 2:
        return 0.0
    m = rolling_baseline_mean(baseline)
    var = sum((x - m) ** 2 for x in history) / len(history)
    return var ** 0.5


def rolling_baseline_z(baseline, value):
    history = baseline["history"]
    if not history:
        return 0.0
    return (value - rolling_baseline_mean(baseline)) / (rolling_baseline_std(baseline) + 1e-6)


def rolling_baseline_ready(baseline):
    return len(baseline["history"]) >= 3


# ---------- LinUCB ----------

def linucb_new(alpha=1.0, state_path="logging/linucb_state.npz"):
    models = ["fp32", "int8"]
    d = 5
    state = {"alpha": alpha, "models": models, "d": d, "state_path": state_path}

    if os.path.exists(state_path):
        data = np.load(state_path)
        state["A"] = {"fp32": data["A_fp32"], "int8": data["A_int8"]}
        state["b"] = {"fp32": data["b_fp32"], "int8": data["b_int8"]}
        print("LinUCB: loaded previous state.")
    else:
        state["A"] = {m: np.identity(d) for m in models}
        state["b"] = {m: np.zeros(d) for m in models}
        print("LinUCB: starting fresh.")

    return state


def linucb_context(cpu, ram, temp, battery):
    return np.array([cpu / 100.0, ram / 100.0, temp / 100.0, battery / 100.0, 1.0])


def linucb_choose(state, cpu, ram, temp, battery):
    x = linucb_context(cpu, ram, temp, battery)
    scores = {}
    for model in state["models"]:
        A_inv = np.linalg.inv(state["A"][model])
        theta = A_inv @ state["b"][model]
        expected = theta @ x
        uncertainty = state["alpha"] * np.sqrt(x @ A_inv @ x)
        scores[model] = expected + uncertainty
    chosen = max(scores, key=scores.get)
    return chosen, scores


def linucb_update(state, model, cpu, ram, temp, battery, reward, confidence=None):
    x = linucb_context(cpu, ram, temp, battery)
    state["A"][model] += np.outer(x, x)
    state["b"][model] += reward * x
    linucb_save(state)


def linucb_save(state):
    np.savez(state["state_path"],
              A_fp32=state["A"]["fp32"], A_int8=state["A"]["int8"],
              b_fp32=state["b"]["fp32"], b_int8=state["b"]["int8"])


# ---------- Epsilon Greedy ----------

def epsilon_greedy_new(epsilon=0.1, state_path="logging/egreedy_state.npz"):
    models = ["fp32", "int8"]
    state = {"epsilon": epsilon, "models": models, "state_path": state_path}

    if os.path.exists(state_path):
        data = np.load(state_path)
        state["counts"] = {m: float(data[f"count_{m}"]) for m in models}
        state["totals"] = {m: float(data[f"total_{m}"]) for m in models}
        print("EpsilonGreedy: loaded previous state.")
    else:
        state["counts"] = {m: 0.0 for m in models}
        state["totals"] = {m: 0.0 for m in models}
        print("EpsilonGreedy: starting fresh.")

    return state


def epsilon_greedy_avg_reward(state, model):
    if state["counts"][model] == 0:
        return 0.0
    return state["totals"][model] / state["counts"][model]


def epsilon_greedy_choose(state, cpu, ram, temp, battery):
    if np.random.random() < state["epsilon"]:
        chosen = np.random.choice(state["models"])
    else:
        chosen = max(state["models"], key=lambda m: epsilon_greedy_avg_reward(state, m))
    scores = {m: round(epsilon_greedy_avg_reward(state, m), 4) for m in state["models"]}
    return chosen, scores


def epsilon_greedy_update(state, model, cpu, ram, temp, battery, reward, confidence=None):
    state["counts"][model] += 1
    state["totals"][model] += reward
    np.savez(state["state_path"],
              **{f"count_{m}": state["counts"][m] for m in state["models"]},
              **{f"total_{m}": state["totals"][m] for m in state["models"]})


# ---------- Eight Signal Controller ----------

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


def eight_signal_new(alpha=1.0, state_path="logging/eight_signal_linucb_state.npz"):
    return {
        "linucb": linucb_new(alpha=alpha, state_path=state_path),
        "baselines": {
            "cpu_util":         rolling_baseline_new(),
            "cpu_freq_ratio":   rolling_baseline_new(),
            "cpu_trend":        rolling_baseline_new(),
            "mem_pressure":     rolling_baseline_new(),
            "mem_activity":     rolling_baseline_new(),
            "thermal_headroom": rolling_baseline_new(),
            "power_rate":       rolling_baseline_new(),
            "difficulty":       rolling_baseline_new(),
        },
        "prev_cpu": None,
        "prev_battery": None,
        "prev_disk_busy": None,
        "prev_confidence": 0.5,
        "last_source": None,
        "last_votes": None,
        "last_S": 0.0,
        "cpu_freq_current": 0.0,
        "cpu_freq_max": 4000.0,
        "disk_busy": 0,
        "current_model": None,
        "iterations_on_current": 0,
    }


def eight_signal_set_extra_telemetry(state, cpu_freq_current, cpu_freq_max, disk_busy_time):
    state["cpu_freq_current"] = cpu_freq_current
    state["cpu_freq_max"] = cpu_freq_max
    state["disk_busy"] = disk_busy_time


def eight_signal_vote(state, name, current_value, higher_is_worse):
    """Continuous vote in [-1, 1] based on z-score magnitude, instead of a
    hard -1/0/+1 snap. A mildly-off signal now contributes less than a
    wildly-off one. Capped at |z|=2.0 so outliers don't dominate."""
    baseline = state["baselines"][name]
    rolling_baseline_update(baseline, current_value)
    if not rolling_baseline_ready(baseline):
        return 0.0
    z = rolling_baseline_z(baseline, current_value)
    scaled = max(-1.0, min(1.0, z / 2.0))
    return -scaled if higher_is_worse else scaled


def eight_signal_apply_hysteresis(state, desired_model):
    """Asymmetric dwell: switching INTO int8 is always immediate.
    Switching back OUT of int8 to fp32 requires MIN_DWELL iterations of
    stability first, so a brief dip in stress doesn't immediately bounce
    it back to the slower model."""
    if state["current_model"] is None:
        state["current_model"] = desired_model
        state["iterations_on_current"] = 1
        return desired_model

    if desired_model == state["current_model"]:
        state["iterations_on_current"] += 1
        return state["current_model"]

    if desired_model == "int8":
        state["current_model"] = "int8"
        state["iterations_on_current"] = 1
        return "int8"

    if state["iterations_on_current"] >= MIN_DWELL:
        state["current_model"] = "fp32"
        state["iterations_on_current"] = 1
        return "fp32"
    else:
        state["iterations_on_current"] += 1
        return state["current_model"]


def eight_signal_choose(state, cpu, ram, temp, battery):
    v1 = eight_signal_vote(state, "cpu_util", cpu, higher_is_worse=True)
    if cpu >= CPU_ABSOLUTE_THRESHOLD:
        v1 = min(v1, -1.0)

    freq_ratio = state["cpu_freq_current"] / (state["cpu_freq_max"] or 1.0)
    v2 = eight_signal_vote(state, "cpu_freq_ratio", freq_ratio, higher_is_worse=False)

    trend = 0.0 if state["prev_cpu"] is None else (cpu - state["prev_cpu"])
    state["prev_cpu"] = cpu
    v3 = eight_signal_vote(state, "cpu_trend", trend, higher_is_worse=True)

    mem_pressure = ram / 100.0
    v4 = eight_signal_vote(state, "mem_pressure", mem_pressure, higher_is_worse=True)

    activity = 0.0 if state["prev_disk_busy"] is None else max(0, state["disk_busy"] - state["prev_disk_busy"])
    state["prev_disk_busy"] = state["disk_busy"]
    v5 = eight_signal_vote(state, "mem_activity", activity, higher_is_worse=True)

    headroom = THERMAL_LIMIT - temp
    v6 = eight_signal_vote(state, "thermal_headroom", headroom, higher_is_worse=False)
    if headroom <= THERMAL_ABSOLUTE_THRESHOLD:
        v6 = min(v6, -1.0)

    rate = 0.0 if state["prev_battery"] is None else max(0.0, state["prev_battery"] - battery)
    state["prev_battery"] = battery
    v7 = eight_signal_vote(state, "power_rate", rate, higher_is_worse=True)

    difficulty = 1.0 - state["prev_confidence"]
    v8 = eight_signal_vote(state, "difficulty", difficulty, higher_is_worse=False)

    votes = [v1, v2, v3, v4, v5, v6, v7, v8]
    S = sum(votes) / len(votes)
    state["last_votes"] = votes
    state["last_S"] = S

    if S > TAU:
        desired_model, scores, source = "fp32", {"fp32": S, "int8": -S}, "vote_fp32"
    elif S < -TAU:
        desired_model, scores, source = "int8", {"fp32": S, "int8": -S}, "vote_int8"
    else:
        desired_model, scores = linucb_choose(state["linucb"], cpu, ram, temp, battery)
        source = "linucb"

    final_model = eight_signal_apply_hysteresis(state, desired_model)
    state["last_source"] = source

    return final_model, scores


def eight_signal_update(state, model, cpu, ram, temp, battery, reward, confidence=None):
    if confidence is not None:
        state["prev_confidence"] = confidence
    if state["last_source"] == "linucb":
        linucb_update(state["linucb"], model, cpu, ram, temp, battery, reward)


def eight_signal_get_last_source(state):
    return state["last_source"]


def eight_signal_get_last_score(state):
    return state["last_S"]


# ---------- Dispatcher ----------

def get_algo(name="linucb", **kwargs):
    name = name.lower().strip()
    if name == "linucb":
        return {"type": "linucb", "state": linucb_new(**kwargs)}
    elif name in ("egreedy", "epsilon"):
        kwargs.pop("alpha", None)
        return {"type": "egreedy", "state": epsilon_greedy_new(**kwargs)}
    elif name in ("eightsignal", "combo", "final"):
        return {"type": "eightsignal", "state": eight_signal_new(**kwargs)}
    else:
        raise ValueError(f"Unknown algorithm '{name}'. Choose 'linucb', 'egreedy', or 'eightsignal'.")


def choose(algo, cpu, ram, temp, battery):
    if algo["type"] == "linucb":
        return linucb_choose(algo["state"], cpu, ram, temp, battery)
    elif algo["type"] == "egreedy":
        return epsilon_greedy_choose(algo["state"], cpu, ram, temp, battery)
    elif algo["type"] == "eightsignal":
        return eight_signal_choose(algo["state"], cpu, ram, temp, battery)


def update(algo, model, cpu, ram, temp, battery, reward, confidence=None):
    if algo["type"] == "linucb":
        linucb_update(algo["state"], model, cpu, ram, temp, battery, reward, confidence)
    elif algo["type"] == "egreedy":
        epsilon_greedy_update(algo["state"], model, cpu, ram, temp, battery, reward, confidence)
    elif algo["type"] == "eightsignal":
        eight_signal_update(algo["state"], model, cpu, ram, temp, battery, reward, confidence)


def get_last_source(algo):
    if algo["type"] == "eightsignal":
        return eight_signal_get_last_source(algo["state"])
    return None


def get_last_score(algo):
    if algo["type"] == "eightsignal":
        return eight_signal_get_last_score(algo["state"])
    return None


def set_extra_telemetry(algo, cpu_freq_current, cpu_freq_max, disk_busy_time):
    if algo["type"] == "eightsignal":
        eight_signal_set_extra_telemetry(algo["state"], cpu_freq_current, cpu_freq_max, disk_busy_time)