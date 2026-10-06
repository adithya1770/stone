"""EdgeMLBalancer-style epsilon-greedy model selection, for comparison with v2.

EdgeMLBalancer (arXiv 2502.06493) switches between several detection models
with an epsilon-greedy strategy: most of the time it uses the model with the
best running performance score (a mix of CPU usage and prediction
confidence); with probability epsilon it tries a random model.

Our version on the model ladder (the paper gives no exact formula):
  arms   = the same models v2 can use (MobileNetV2, EfficientNet-Lite0..4)
  score  = confidence - lam * max(0, latency - budget) / budget
           i.e. confidence, minus a penalty only when the photo took longer than
           the time budget (resource cost; under CPU contention latency is what
           the contention changes)
  choice = untried arms first; then with prob. epsilon a random arm, otherwise
           the arm with the best running mean score (kept per device state)
The model is chosen BEFORE seeing the photo, like the original controllers.
"""
import random

from .policies import Policy


class EpsilonGreedyLadder(Policy):
    name = "egreedy_ladder"

    def __init__(self, arms, budget_ms=150.0, epsilon=0.1, lam=1.0, seed=0, per_tier=True, name=None):
        self.arms = list(arms)
        self.budget = float(budget_ms)
        self.eps = float(epsilon)
        self.lam = float(lam)
        self.per_tier = per_tier
        self.rng = random.Random(seed)
        self.stats = {}          # tier -> arm -> [n, mean_score]
        if name:
            self.name = name

    def _table(self, ctx):
        key = ctx.get("tier", "idle") if self.per_tier else "all"
        return self.stats.setdefault(key, {a: [0, 0.0] for a in self.arms})

    def score(self, confidence, latency_ms):
        return confidence - self.lam * max(0.0, latency_ms - self.budget) / self.budget

    def first(self, ctx):
        tab = self._table(ctx)
        untried = [a for a in self.arms if tab[a][0] == 0]
        if untried:
            return untried[0]
        if self.rng.random() < self.eps:
            return self.rng.choice(self.arms)
        return max(self.arms, key=lambda a: tab[a][1])

    def observe(self, ctx, results, total_ms):
        r = results[0]
        if r is None:
            return
        tab = self._table(ctx)
        n, m = tab[r["variant"]]
        s = self.score(r["confidence"], total_ms)
        tab[r["variant"]] = [n + 1, m + (s - m) / (n + 1)]

    def state(self):
        return {t: {a: round(v[1], 3) for a, v in tab.items()} for t, tab in self.stats.items()}
