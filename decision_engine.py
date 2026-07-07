

ACTION_FP32 = 0
ACTION_INT8 = 1

HIGH_THRESHOLD = 70
LOW_THRESHOLD = 50
REQUIRED_READINGS = 10


class DecisionEngine:

    def __init__(self):

        self.current_action = ACTION_FP32

        self.high_count = 0
        self.low_count = 0

    def choose_action(self, telemetry):

        cpu_ema = telemetry["cpu_ema"]

        if cpu_ema > HIGH_THRESHOLD:

            self.high_count += 1
            self.low_count = 0

        elif cpu_ema < LOW_THRESHOLD:

            self.low_count += 1
            self.high_count = 0

        else:

            self.high_count = 0
            self.low_count = 0

        if self.high_count >= REQUIRED_READINGS:

            self.current_action = ACTION_INT8

        if self.low_count >= REQUIRED_READINGS:

            self.current_action = ACTION_FP32

        return self.current_action
    
if __name__ == "__main__":

    engine = DecisionEngine()

    for _ in range(12):
        print(
            engine.choose_action(
                {"cpu_ema": 80}
            )
        )