import numpy as np


class LinUCB:
    def __init__(self, n_actions: int, n_features: int, alpha: float = 1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha  # exploration strength: higher alpha = more exploration

        self.A = [np.identity(n_features) for _ in range(n_actions)]
        self.b = [np.zeros(n_features) for _ in range(n_actions)]

    def select_action(self, context: np.ndarray) -> int:
        p_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            expected_reward = theta @ context
            exploration_bonus = self.alpha * np.sqrt(context @ A_inv @ context)
            p_values[a] = expected_reward + exploration_bonus
        return int(np.argmax(p_values))
    def update(self, action: int, context: np.ndarray, reward: float):
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context
    def theta(self, action: int) -> np.ndarray:
        return np.linalg.inv(self.A[action]) @ self.b[action]