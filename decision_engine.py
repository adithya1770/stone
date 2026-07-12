
import time
from linucb import LinUCB
from context import build_context

class DecisionEngine:
    def __init__(self, linucb: LinUCB, min_mode_duration=5):
        self.linucb = linucb
        self.min_mode_duration = min_mode_duration
        self.current_action = None
        self.last_switch_time = 0
        self.prev_latency = 0.0

    def decide(self, telemetry_snapshot):
        now = time.time()
        context = build_context(telemetry_snapshot, self.prev_latency)
        proposed = self.linucb.select_action(context)

        if self.current_action is None:
            self.current_action = proposed
            self.last_switch_time = now
        else:
            time_since_switch = now - self.last_switch_time
            if proposed != self.current_action and time_since_switch >= self.min_mode_duration:
                self.current_action = proposed
                self.last_switch_time = now

        return self.current_action, context  # main.py needs context to call learn() later

    def learn(self, action, context, reward, latency_ms):
        self.linucb.update(action, context, reward)
        self.prev_latency = latency_ms