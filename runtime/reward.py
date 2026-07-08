def calculate_reward(confidence, latency_ms):
    latency_penalty = latency_ms / 100
    reward = confidence - latency_penalty
    return round(reward, 4)