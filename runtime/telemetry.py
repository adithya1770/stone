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

    try:
        battery_info = psutil.sensors_battery()
        battery = battery_info.percent if battery_info is not None else 100.0
    except AttributeError:
        battery = 100.0

    try:
        freq = psutil.cpu_freq()
        cpu_freq_current = freq.current if freq else 0.0
        cpu_freq_max = freq.max if freq and freq.max else 4000.0
    except Exception:
        cpu_freq_current = 0.0
        cpu_freq_max = 4000.0

    try:
        io = psutil.disk_io_counters()
        disk_busy_time = io.busy_time if io else 0
    except Exception:
        disk_busy_time = 0

    return {
        "cpu":              cpu,
        "ram":              round(ram, 2),
        "temperature":      temperature,
        "battery":          round(battery, 2),
        "cpu_per_core":     [round(c, 1) for c in per_core],
        "cpu_freq_current": cpu_freq_current,
        "cpu_freq_max":     cpu_freq_max,
        "disk_busy_time":   disk_busy_time,
    }