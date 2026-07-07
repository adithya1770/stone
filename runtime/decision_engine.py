HIGH_THRESHOLD = 70
LOW_THRESHOLD = 50
REQUIRED_READINGS = 10


def health_score(cpu, ram, temp):
    return (0.5 * cpu) + (0.2 * ram) + (0.3 * temp)


def decide(cpu, ram, temp, state):
    health = health_score(cpu, ram, temp)

    if health > HIGH_THRESHOLD:
        state["high_count"] += 1
        state["low_count"] = 0
    elif health < LOW_THRESHOLD:
        state["low_count"] += 1
        state["high_count"] = 0
    else:
        state["high_count"] = 0
        state["low_count"] = 0

    if state["high_count"] >= REQUIRED_READINGS:
        state["current_model"] = "int8"
        state["high_count"] = 0

    if state["low_count"] >= REQUIRED_READINGS:
        state["current_model"] = "fp32"
        state["low_count"] = 0

    return {
        "model": state["current_model"],
        "health_score": round(health, 2)
    }