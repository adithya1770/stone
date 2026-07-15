import math


def calculate_reward(
    confidence,
    latency_ms,
    cpu,
    ram,
    temperature,
):
    health = (
        0.5 * cpu +
        0.2 * ram +
        0.3 * temperature
    )

    stress = max(0.0, min(1.0, health / 100.0))

    accuracy_weight = 0.8 - 0.5 * stress
    latency_weight  = 0.2 + 0.5 * stress

    accuracy_weight /= (accuracy_weight + latency_weight)
    latency_weight  /= (accuracy_weight + latency_weight)

    confidence_score = confidence

    latency_score = max(
        0,
        1 - latency_ms / 80
    )

    reward = (
        accuracy_weight * confidence_score +
        latency_weight * latency_score
    )

    return round(reward, 4)