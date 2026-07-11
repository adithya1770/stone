
import psutil
import time


class Telemetry:

    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        self.cpu_ema = None
        self.mem_ema = None

    def _read_cpu(self) -> float:
        # interval=0.1 gives a real sampled reading instead of an instant 0.0
        return psutil.cpu_percent(interval=0.1)

    def _read_memory(self) -> float:
        return psutil.virtual_memory().percent

    def _read_temperature(self):
        try:
            temps = psutil.sensors_temperatures()
        except (AttributeError, NotImplementedError):
            return None

        if not temps:
            return None
        first_sensor = next(iter(temps.values()))
        if not first_sensor:
            return None

        return first_sensor[0].current

    def _update_ema(self, current: float, previous_ema):
        if previous_ema is None:
            return current  # first reading
        return self.ema_alpha * current + (1 - self.ema_alpha) * previous_ema

    def read(self) -> dict:
        """
        Takes one telemetry snapshot, updates internal EMA state,
        and returns both raw and smoothed values.
        """
        cpu_raw = self._read_cpu()
        mem_raw = self._read_memory()
        temp_raw = self._read_temperature()

        self.cpu_ema = self._update_ema(cpu_raw, self.cpu_ema)
        self.mem_ema = self._update_ema(mem_raw, self.mem_ema)

        return {
            "cpu_raw": cpu_raw,
            "cpu_ema": self.cpu_ema,
            "mem_raw": mem_raw,
            "mem_ema": self.mem_ema,
            "temp": temp_raw,  # None if no sensor available
        }


if __name__ == "__main__":
    t = Telemetry()
    print("Reading telemetry every second. Ctrl+C to stop.")
    try:
        while True:
            snapshot = t.read()
            print(snapshot)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")
