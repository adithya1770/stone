import psutil

def get_telemetry():
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory().percent

    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            temperature = temps["coretemp"][0].current
        elif "cpu_thermal" in temps:
            temperature = temps["cpu_thermal"][0].current
        else:
            temperature = None
    except AttributeError:
        temperature = None

    return {
        "cpu_percent": cpu,
        "ram_percent": ram,
        "temperature_c": temperature
    }