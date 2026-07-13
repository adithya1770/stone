import time

class DecisionEngine:
    """
    Unified wrapper: hysteresis logic stays identical no matter which policy
    is plugged in. Baseline, rule-based, and both LinUCB conditions all run
    through the exact same decision-timing code -- any measured difference
    between them is due to the policy itself, not the harness.
    """
    def __init__(self, policy, min_mode_duration=5):
        self.policy = policy
        self.min_mode_duration = min_mode_duration
        self.current_action = None
        self.last_switch_time = 0
        self.prev_latency = 0.0

    def decide(self, telemetry_snapshot):
        now = time.time()
        proposed, context = self.policy.select_action(telemetry_snapshot, self.prev_latency)

        if self.current_action is None:
            self.current_action = proposed
            self.last_switch_time = now
        else:
            time_since_switch = now - self.last_switch_time
            if proposed != self.current_action and time_since_switch >= self.min_mode_duration:
                self.current_action = proposed
                self.last_switch_time = now

        return self.current_action, context

    def learn(self, action, context, reward, latency_ms):
        self.policy.update(action, context, reward, latency_ms)
        self.prev_latency = latency_ms