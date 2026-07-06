import psutil


class TelemetryMonitor:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.cpu_ema = None

    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=1)

    def get_memory_usage(self):
        return psutil.virtual_memory().percent

    def get_temperature(self):
        temps = psutil.sensors_temperatures()

        if not temps:
            return None

        for sensor_name, entries in temps.items():
            if entries:
                return entries[0].current

        return None

    def update_cpu_ema(self, cpu):

        if self.cpu_ema is None:
            self.cpu_ema = cpu

        else:
            self.cpu_ema = (
                self.alpha * cpu
                + (1 - self.alpha) * self.cpu_ema
            )

        return self.cpu_ema

    def get_telemetry(self):

        cpu = self.get_cpu_usage()

        return {
            "cpu": cpu,
            "cpu_ema": round(self.update_cpu_ema(cpu),2),
            "memory": self.get_memory_usage(),
            "temperature": self.get_temperature()
        }
    
