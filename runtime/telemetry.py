import psutil


def get_telemetry():
    per_core = psutil.cpu_percent(percpu=True, interval=0.5)

    cpu_avg = sum(per_core) / len(per_core)
    cpu_max = max(per_core)
    cpu = round(max(cpu_avg, cpu_max), 2)

    ram = psutil.virtual_memory().percent

    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            temperature = temps["coretemp"][0].current
        elif "cpu_thermal" in temps:
            temperature = temps["cpu_thermal"][0].current
        else:
            temperature = 40
    except AttributeError:
        temperature = 40

    return {
        "cpu":         cpu,
        "ram":         round(ram, 2),
        "temperature": temperature,
        "cpu_per_core": [round(c, 1) for c in per_core]
    }