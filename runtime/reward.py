def calculate_reward(confidence, latency_ms, target_ms=20):
    if latency_ms <= target_ms:
        latency_penalty = 0
    else:
        latency_penalty = (latency_ms - target_ms) / 50

    return round(confidence - latency_penalty, 4)