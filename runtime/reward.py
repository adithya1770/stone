def calculate_reward(confidence, latency_ms):
    latency_penalty = latency_ms / 50
    reward = confidence - latency_penalty
    return round(reward, 4)