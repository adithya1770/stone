from heuristics import choose_action as heuristic_choose_action
from linucb import LinUCB
from context import build_context


class FixedPolicy:
    """Always returns the same action. Used for the baseline (fixed FP32) condition."""
    def __init__(self, action):
        self.action = action

    def select_action(self, telemetry_snapshot, prev_latency):
        return self.action, None

    def update(self, action, context, reward, latency_ms):
        pass  # baseline doesn't learn


class HeuristicPolicy:
    """Wraps the existing rule-based heuristic. Doesn't need context."""
    def select_action(self, telemetry_snapshot, prev_latency):
        return heuristic_choose_action(telemetry_snapshot), None

    def update(self, action, context, reward, latency_ms):
        pass  # heuristic doesn't learn


class LinUCBPolicy:
    """Wraps a LinUCB instance. Works for both cold (fresh) and warm (bootstrapped)."""
    def __init__(self, linucb: LinUCB):
        self.linucb = linucb

    def select_action(self, telemetry_snapshot, prev_latency):
        context = build_context(telemetry_snapshot, prev_latency)
        return self.linucb.select_action(context), context

    def update(self, action, context, reward, latency_ms):
        if context is not None:
            self.linucb.update(action, context, reward)