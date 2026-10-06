"""CPU stress helper: stress-ng if installed (same tool as the original
experiments), otherwise pure-Python busy-loop processes on every core."""
import multiprocessing as mp
import shutil
import subprocess
import time


def _burn(stop):
    x = 0
    while not stop.is_set():
        for _ in range(100000):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF


class Stress:
    def __init__(self, seconds=600, prefer="auto"):
        self.seconds = seconds
        self.prefer = prefer
        self.kind = None
        self._proc = None
        self._stop = None
        self._workers = []

    def start(self, settle=2.0):
        if self.prefer in ("auto", "stress-ng") and shutil.which("stress-ng"):
            self.kind = "stress-ng"
            self._proc = subprocess.Popen(["stress-ng", "--cpu", "0", "--timeout", f"{self.seconds}s"],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            self.kind = "python-burn"
            self._stop = mp.Event()
            self._workers = [mp.Process(target=_burn, args=(self._stop,), daemon=True)
                             for _ in range(mp.cpu_count())]
            for w in self._workers:
                w.start()
        time.sleep(settle)
        return self

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._stop is not None:
            self._stop.set()
            for w in self._workers:
                w.join(timeout=5)
                if w.is_alive():
                    w.terminate()

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()
