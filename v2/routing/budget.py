"""Latency-budget controller for the escalation threshold.

Rule: escalate an image to the accurate model when the cheap model's
uncertainty score is below theta. The score is one of (see SIGNALS):
  margin      p1 - p2                     in [0, 1]
  confidence  p1                          in [0, 1]
  entropy     -entropy(probs)             in [-ln 1000, 0]
Which one is used is chosen offline per tier (bench/replay_eval.py).

Online update after each request (stochastic approximation / dual ascent):
    theta <- clip(theta + eta * (budget - latency) / budget)
  - running over budget  -> theta falls -> fewer escalations
  - running under budget -> theta rises -> more escalations
At equilibrium the mean latency sits at the budget. It needs only measured
latency, never ground-truth labels, so it works in the live server.

Under CPU stress every model gets slower, so the same budget automatically
means fewer escalations: resource-awareness emerges from the constraint.

Optional hard deadline: never escalate if first + second stage are
expected to exceed `deadline_ms` (protects tail latency).
"""


import math

SIGNALS = {"margin": (0.0, 1.0), "confidence": (0.0, 1.0), "entropy": (-math.log(1000), 0.0)}


def signal_score(result, signal):
    """Higher = more confident, for every signal."""
    return -result["entropy"] if signal == "entropy" else result[signal]


class BudgetController:
    def __init__(self, budget_ms, theta0=0.3, eta=0.02, theta_min=0.0,
                 theta_max=1.0, deadline_ms=None, adapt=True):
        self.budget_ms = float(budget_ms)
        self.theta = float(theta0)
        self.eta = eta
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.deadline_ms = deadline_ms
        self.adapt = adapt
        self.n = 0
        self.escalations = 0

    def should_escalate(self, margin, est_first_ms=None, est_second_ms=None):
        if self.deadline_ms is not None and est_first_ms is not None and est_second_ms is not None:
            if est_first_ms + est_second_ms > self.deadline_ms:
                return False
        return margin < self.theta

    def observe(self, latency_ms, escalated=False):
        self.n += 1
        self.escalations += int(bool(escalated))
        if self.adapt and self.budget_ms > 0:
            span = self.theta_max - self.theta_min
            self.theta += self.eta * span * (self.budget_ms - latency_ms) / self.budget_ms
            self.theta = min(self.theta_max, max(self.theta_min, self.theta))
        return self.theta


class LatencyTracker:
    """EMA of observed latency per (tier, variant), seeded from profiling."""

    def __init__(self, seed=None, alpha=0.1):
        self.alpha = alpha
        self.est = {}
        for tier, d in (seed or {}).items():
            for v, ms in d.items():
                self.est[(tier, v)] = float(ms)

    def get(self, tier, variant, default=None):
        return self.est.get((tier, variant), default)

    def update(self, tier, variant, ms):
        k = (tier, variant)
        self.est[k] = ms if k not in self.est else (1 - self.alpha) * self.est[k] + self.alpha * ms
