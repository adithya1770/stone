import psutil

class TelemetryMonitor:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.cpu_ema = None

    def update_ema(self, cpu):
        if self.cpu_ema is None:
            self.cpu_ema = cpu
        else:
            self.cpu_ema = self.alpha * cpu + (1 - self.alpha) * self.cpu_ema

        return round(self.cpu_ema, 2)

    def get_temperature(self):
        try:
            temps = psutil.sensors_temperatures()

            if "coretemp" in temps:
                return temps["coretemp"][0].current

            elif "cpu_thermal" in temps:
                return temps["cpu_thermal"][0].current

        except AttributeError:
            pass

        return None
        

    def get_telemetry(self):
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent

        return {
            "cpu": cpu,
            "cpu_ema": self.update_ema(cpu),
            "ram": ram,
            "temperature": self.get_temperature()
        }