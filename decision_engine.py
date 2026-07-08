import time

ACTION_FP32 = 0
ACTION_INT8 = 1
CPU_THRESHOLD = 30

class DecisionEngine:

    def __init__(self, min_mode_duration=5):
        self.min_mode_duration = min_mode_duration
        self.current_action = None
        self.last_switch_time = 0

    def _choose_action(self, telemetry):
        if telemetry["cpu_ema"] > CPU_THRESHOLD:
            return ACTION_INT8
        return ACTION_FP32
    
    def decide(self, telemetry):
        now = time.time()
        proposed = self._choose_action(telemetry)
        if self.current_action is None:
            self.current_action = proposed
            self.last_switch_time = now
            return self.current_action
        time_since_switch = now - self.last_switch_time
        if (
            proposed != self.current_action
            and time_since_switch >= self.min_mode_duration
        ):
            self.current_action = proposed
            self.last_switch_time = now
        return self.current_action