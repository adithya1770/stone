# heuristics.py

ACTION_FP32 = 0
ACTION_INT8 = 1
CPU_THRESHOLD = 30

def choose_action(telemetry_snapshot):
    if telemetry_snapshot["cpu_ema"] > CPU_THRESHOLD:
        return ACTION_INT8
    return ACTION_FP32