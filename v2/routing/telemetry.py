"""Non-blocking telemetry for STONE v2.

The original runtime/telemetry.get_telemetry() blocks for 0.5 s on every
call (psutil.cpu_percent(interval=0.5)). web_demo/server.py calls it once
per request, so every request waits ~500 ms before inference even starts.

BackgroundTelemetry samples in a daemon thread and snapshot() returns
instantly. It also reports:
  - cpu_external: system CPU load NOT caused by this process. With a
    background sampler, our own inference would otherwise look like
    "stress" and push the router toward cheaper models.
  - throttled: Raspberry Pi firmware throttle flags (under-voltage,
    frequency cap, throttling, soft temperature limit), when available.
"""
import os
import subprocess
import threading
import time

import psutil

_THROTTLE_SYSFS = "/sys/devices/platform/soc/soc:firmware/get_throttled"


def read_temperature():
    try:
        temps = psutil.sensors_temperatures()
        for key in ("cpu_thermal", "coretemp", "k10temp", "soc_thermal"):
            if key in temps and temps[key]:
                return float(temps[key][0].current)
    except (AttributeError, OSError):
        pass
    return None


def read_throttled():
    """Return the Pi throttle bitmask (int) or None if not on a Pi."""
    try:
        with open(_THROTTLE_SYSFS) as f:
            return int(f.read().strip(), 16)
    except (OSError, ValueError):
        pass
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=1).stdout
        return int(out.strip().split("=")[1], 16)
    except Exception:
        return None


class BackgroundTelemetry:
    def __init__(self, interval=0.25, throttle_every=2.0):
        self.interval = interval
        self.throttle_every = throttle_every
        self._proc = psutil.Process()
        self._ncpu = psutil.cpu_count() or 1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._state = {}
        self._thread = None
        psutil.cpu_percent(percpu=True, interval=None)  # prime counters
        self._proc.cpu_percent(interval=None)
        self._sample(force_throttle=True)

    def _sample(self, force_throttle=False):
        per_core = psutil.cpu_percent(percpu=True, interval=None) or [0.0]
        own = self._proc.cpu_percent(interval=None)  # % of ONE core
        cpu_avg = sum(per_core) / len(per_core)
        cpu_external = max(0.0, (cpu_avg * self._ncpu - own) / self._ncpu)
        freq = None
        try:
            freq = psutil.cpu_freq()
        except Exception:
            pass
        now = time.time()
        with self._lock:
            prev_thr = self._state.get("throttled")
            last_thr_t = self._state.get("_thr_t", 0.0)
        thr = prev_thr
        if force_throttle or now - last_thr_t >= self.throttle_every:
            thr = read_throttled()
            last_thr_t = now
        temp = read_temperature()
        state = {
            "t": now,
            "cpu": round(max(cpu_avg, max(per_core)), 2),  # original definition
            "cpu_avg": round(cpu_avg, 2),
            "cpu_external": round(cpu_external, 2),
            "own_cpu_pct": round(own, 2),
            "ram": round(psutil.virtual_memory().percent, 2),
            "temperature": temp if temp is not None else 40.0,  # original fallback
            "temperature_known": temp is not None,
            "battery": 100.0,
            "cpu_freq_current": float(freq.current) if freq else 0.0,
            "cpu_freq_max": float(freq.max) if freq and freq.max else 0.0,
            "throttled": thr,
            "throttled_now": bool(thr & 0xF) if thr is not None else False,
            "_thr_t": last_thr_t,
        }
        with self._lock:
            self._state = state

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def snapshot(self):
        with self._lock:
            return {k: v for k, v in self._state.items() if not k.startswith("_")}

    def process_cpu_seconds(self):
        ct = self._proc.cpu_times()
        return ct.user + ct.system

    def rss_mb(self):
        return self._proc.memory_info().rss / 1e6
