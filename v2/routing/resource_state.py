"""Resource awareness: turn telemetry into a device tier.

Tiers
  idle      - headroom available: cheap first look, escalate freely
  stressed  - CPU contention / heat / frequency drop: different ladder,
              and the latency budget escalates less automatically
  critical  - hard thermal limit (>= 85 C, where the Pi 4 firmware throttles)
              or firmware "currently throttled" flag: the stressed ladder's
              first model, never escalate (protect the device)

Two modes:
  'simple'      transparent absolute thresholds (default)
  'eightsignal' reuses the original EightSignalController's vote score S
                (imported read-only from runtime/decision_engine.py), so
                v2 builds directly on the original resource-sensing work.

Hysteresis follows the original asymmetric-dwell idea: moving to a more
constrained tier is immediate, relaxing needs `dwell` calm readings.
"""
import os

TIERS = ("idle", "stressed", "critical")
_RANK = {t: i for i, t in enumerate(TIERS)}


class ResourceMonitor:
    def __init__(self, mode="simple", cpu_hi=60.0, temp_hi=75.0,
                 temp_critical=85.0, freq_drop=0.8, dwell=3,
                 eightsignal_state_path=None):
        self.mode = mode
        self.cpu_hi = cpu_hi
        self.temp_hi = temp_hi
        self.temp_critical = temp_critical
        self.freq_drop = freq_drop
        self.dwell = dwell
        self.tier = "idle"
        self._calm = 0
        self.last_raw = "idle"
        self.last_score = None
        if mode == "eightsignal":
            from runtime import decision_engine as de  # original, read-only
            from v2.common.paths import results_path
            self._de = de
            path = eightsignal_state_path or results_path("state", "eightsignal_tier_linucb.npz")
            if os.path.exists(path):
                os.remove(path)  # fresh; lives under v2/results_v2 only
            self._es = de.eight_signal_new(alpha=1.0, state_path=path)

    # -- raw tier from one telemetry snapshot -----------------------------
    @staticmethod
    def _hard_throttled(t):
        """Pi firmware bit 2 = 'currently throttled'. Under-voltage (bit 0) or a
        soft frequency cap alone do not trigger critical: the latency budget
        already absorbs the slow-down."""
        thr = t.get("throttled")
        return isinstance(thr, int) and bool(thr & 0x4)

    def _raw_simple(self, t):
        temp = t.get("temperature", 40.0)
        if self._hard_throttled(t) or (t.get("temperature_known", True) and temp >= self.temp_critical):
            return "critical"
        cpu = t.get("cpu_external", t.get("cpu", 0.0))
        fmax = t.get("cpu_freq_max") or 0.0
        fcur = t.get("cpu_freq_current") or 0.0
        freq_low = fmax > 0 and fcur > 0 and (fcur / fmax) < self.freq_drop
        if cpu >= self.cpu_hi or temp >= self.temp_hi or freq_low:
            return "stressed"
        return "idle"

    def _raw_eightsignal(self, t):
        if self._hard_throttled(t) or t.get("temperature", 40.0) >= self.temp_critical:
            return "critical"
        de, st = self._de, self._es
        de.eight_signal_set_extra_telemetry(st, t.get("cpu_freq_current", 0.0),
                                            t.get("cpu_freq_max") or 4000.0, 0)
        de.eight_signal_choose(st, t.get("cpu", 0.0), t.get("ram", 0.0),
                               t.get("temperature", 40.0), t.get("battery", 100.0))
        self.last_score = st["last_S"]
        return "stressed" if (st["last_S"] < -de.TAU or t.get("cpu", 0.0) >= de.CPU_ABSOLUTE_THRESHOLD) else "idle"

    def update(self, t):
        raw = self._raw_eightsignal(t) if self.mode == "eightsignal" else self._raw_simple(t)
        self.last_raw = raw
        if _RANK[raw] >= _RANK[self.tier]:
            self.tier = raw          # tighten immediately
            self._calm = 0
        else:
            self._calm += 1          # relax only after `dwell` calm readings
            if self._calm >= self.dwell:
                self.tier = raw
                self._calm = 0
        return self.tier
