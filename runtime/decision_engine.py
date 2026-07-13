import numpy as np
import os


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
        "model":        state["current_model"],
        "health_score": round(health, 2)
    }


class _BanditBase:
    def __init__(self, state_path):
        self.models = ["fp32", "int8"]
        self.d = 3
        self.state_path = state_path

        if os.path.exists(state_path):
            self._load()
            print(f"{self.__class__.__name__}: loaded previous state.")
        else:
            self._init_fresh()
            print(f"{self.__class__.__name__}: starting fresh.")

    def _init_fresh(self):
        self.A = {m: np.identity(self.d) for m in self.models}
        self.b = {m: np.zeros(self.d)    for m in self.models}

    def _context(self, cpu, ram, temp):
        return np.array([cpu / 100.0, ram / 100.0, temp / 100.0])

    def update(self, model, cpu, ram, temp, reward):
        x = self._context(cpu, ram, temp)
        self.A[model] += np.outer(x, x)
        self.b[model] += reward * x
        self._save()

    def _save(self):
        np.savez(
            self.state_path,
            A_fp32=self.A["fp32"],
            A_int8=self.A["int8"],
            b_fp32=self.b["fp32"],
            b_int8=self.b["int8"]
        )

    def _load(self):
        data = np.load(self.state_path)
        self.A = {
            "fp32": data["A_fp32"],
            "int8": data["A_int8"]
        }
        self.b = {
            "fp32": data["b_fp32"],
            "int8": data["b_int8"]
        }


class LinUCB(_BanditBase):
    def __init__(self, alpha=1.0,
                 state_path="logging/linucb_state.npz"):
        self.alpha = alpha
        super().__init__(state_path)

    def choose(self, cpu, ram, temp):
        x = self._context(cpu, ram, temp)
        scores = {}

        for model in self.models:
            A_inv       = np.linalg.inv(self.A[model])
            theta       = A_inv @ self.b[model]
            expected    = theta @ x
            uncertainty = self.alpha * np.sqrt(x @ A_inv @ x)
            scores[model] = expected + uncertainty

        chosen = max(scores, key=scores.get)
        return chosen, scores


class ThompsonSampling(_BanditBase):
    def __init__(self, state_path="logging/thompson_state.npz"):
        super().__init__(state_path)

    def choose(self, cpu, ram, temp):
        x = self._context(cpu, ram, temp)
        scores = {}

        for model in self.models:
            A_inv        = np.linalg.inv(self.A[model])
            theta_mean   = A_inv @ self.b[model]
            theta_sample = np.random.multivariate_normal(
                theta_mean, A_inv
            )
            scores[model] = theta_sample @ x

        chosen = max(scores, key=scores.get)
        return chosen, scores


class SlidingLinUCB:
    def __init__(self, alpha=1.0, window=50,
                 state_path="logging/sliding_linucb_state.npz"):
        self.alpha      = alpha
        self.window     = window
        self.models     = ["fp32", "int8"]
        self.d          = 3
        self.state_path = state_path
        self.history    = []

        if os.path.exists(state_path):
            self._load()
            print("SlidingLinUCB: loaded previous state.")
        else:
            self.A = {m: np.identity(self.d) for m in self.models}
            self.b = {m: np.zeros(self.d)    for m in self.models}
            print("SlidingLinUCB: starting fresh.")

    def _context(self, cpu, ram, temp):
        return np.array([cpu / 100.0, ram / 100.0, temp / 100.0])

    def choose(self, cpu, ram, temp):
        x = self._context(cpu, ram, temp)
        scores = {}

        for model in self.models:
            A_inv       = np.linalg.inv(self.A[model])
            theta       = A_inv @ self.b[model]
            expected    = theta @ x
            uncertainty = self.alpha * np.sqrt(x @ A_inv @ x)
            scores[model] = expected + uncertainty

        chosen = max(scores, key=scores.get)
        return chosen, scores

    def update(self, model, cpu, ram, temp, reward):
        x = self._context(cpu, ram, temp)

        self.history.append((x, model, reward))

        if len(self.history) > self.window:
            self.history.pop(0)

        self.A = {m: np.identity(self.d) for m in self.models}
        self.b = {m: np.zeros(self.d)    for m in self.models}

        for hist_x, hist_model, hist_reward in self.history:
            self.A[hist_model] += np.outer(hist_x, hist_x)
            self.b[hist_model] += hist_reward * hist_x

        self._save()

    def _save(self):
        history_x      = np.array([h[0] for h in self.history])
        history_models = np.array([h[1] for h in self.history])
        history_rewards = np.array([h[2] for h in self.history])

        np.savez(
            self.state_path,
            history_x=history_x,
            history_models=history_models,
            history_rewards=history_rewards,
            window=np.array([self.window])
        )

    def _load(self):
        data = np.load(self.state_path, allow_pickle=True)

        history_x       = data["history_x"]
        history_models  = data["history_models"]
        history_rewards = data["history_rewards"]

        self.history = [
            (history_x[i], str(history_models[i]), float(history_rewards[i]))
            for i in range(len(history_x))
        ]

        self.A = {m: np.identity(self.d) for m in self.models}
        self.b = {m: np.zeros(self.d)    for m in self.models}

        for hist_x, hist_model, hist_reward in self.history:
            self.A[hist_model] += np.outer(hist_x, hist_x)
            self.b[hist_model] += hist_reward * hist_x


class EpsilonGreedy:
    def __init__(self, epsilon=0.1,
                 state_path="logging/egreedy_state.npz"):
        self.epsilon    = epsilon
        self.models     = ["fp32", "int8"]
        self.state_path = state_path

        if os.path.exists(state_path):
            data = np.load(state_path)
            self.counts = {m: float(data[f"count_{m}"])
                           for m in self.models}
            self.totals = {m: float(data[f"total_{m}"])
                           for m in self.models}
            print("EpsilonGreedy: loaded previous state.")
        else:
            self.counts = {m: 0.0 for m in self.models}
            self.totals = {m: 0.0 for m in self.models}
            print("EpsilonGreedy: starting fresh.")

    def _avg_reward(self, model):
        if self.counts[model] == 0:
            return 0.0
        return self.totals[model] / self.counts[model]

    def choose(self, cpu, ram, temp):
        if np.random.random() < self.epsilon:
            chosen = np.random.choice(self.models)
        else:
            chosen = max(self.models, key=self._avg_reward)

        scores = {m: round(self._avg_reward(m), 4)
                  for m in self.models}
        return chosen, scores

    def update(self, model, cpu, ram, temp, reward):
        self.counts[model] += 1
        self.totals[model] += reward
        np.savez(
            self.state_path,
            **{f"count_{m}": self.counts[m] for m in self.models},
            **{f"total_{m}": self.totals[m] for m in self.models}
        )


def get_algo(name="linucb", **kwargs):
    name = name.lower().strip()
    if name == "linucb":
        return LinUCB(**kwargs)
    elif name in ("thompson", "ts"):
        kwargs.pop("alpha", None)
        return ThompsonSampling(**kwargs)
    elif name in ("sliding", "slidinglinucb"):
        return SlidingLinUCB(**kwargs)
    elif name in ("egreedy", "epsilon"):
        kwargs.pop("alpha", None)
        return EpsilonGreedy(**kwargs)
    else:
        raise ValueError(
            f"Unknown algorithm '{name}'. "
            f"Choose 'linucb', 'thompson', 'sliding', or 'egreedy'."
        )