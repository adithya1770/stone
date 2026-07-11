import numpy as np

CPU_MAX = 100.0
MEM_MAX = 100.0
TEMP_MAX = 90.0
LATENCY_MAX = 50.0  # ms


def build_context(telemetry_snapshot: dict, prev_latency_ms: float = 0.0) -> np.ndarray:
    cpu_norm = telemetry_snapshot["cpu_ema"] / CPU_MAX
    mem_norm = telemetry_snapshot["mem_ema"] / MEM_MAX
    temp_val = telemetry_snapshot["temp"] if telemetry_snapshot["temp"] is not None else 0.0
    temp_norm = temp_val / TEMP_MAX
    latency_norm = prev_latency_ms / LATENCY_MAX

    return np.array([1.0, cpu_norm, mem_norm, temp_norm, latency_norm])

N_FEATURES = 5  