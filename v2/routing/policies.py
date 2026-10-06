"""Routing policies for STONE v2.

The SAME classes are used by the live runtime (run_suite_v2.py,
server_v2.py) and by offline replay (bench/replay_eval.py), so replay
results describe exactly the code that runs on the Pi.

Protocol per request:
    v1 = policy.first(ctx)             # ctx = {"tier": ..., telemetry...}
    r1 = run(v1)
    v2 = policy.second(ctx, r1)        # None -> answer with r1
    r2 = run(v2) if v2 else None
    policy.observe(ctx, [r1, r2], total_latency_ms)
"""
import json
import random

from .budget import SIGNALS, BudgetController, LatencyTracker, signal_score


class Policy:
    name = "policy"

    def first(self, ctx):
        raise NotImplementedError

    def second(self, ctx, r1):
        return None

    def observe(self, ctx, results, total_ms):
        pass

    def state(self):
        return {}


class Static(Policy):
    """Always the same variant (Baseline = fp32_224, AlwaysINT8 = int8_224)."""

    def __init__(self, variant, name=None):
        self.variant = variant
        self.name = name or f"static_{variant}"

    def first(self, ctx):
        return self.variant


class RandomMix(Policy):
    """Weighted coin flip between two variants. The fair baseline: any
    smart router must beat this at the same mix rate."""

    def __init__(self, a, b, p_b, seed=0, name=None):
        self.a, self.b, self.p_b = a, b, p_b
        self.rng = random.Random(seed)
        self.name = name or f"mix_{a}_{b}_{p_b:.2f}"

    def first(self, ctx):
        return self.b if self.rng.random() < self.p_b else self.a


class FixedCascade(Policy):
    """Input-aware only: cheap model first, escalate if margin < theta.
    No resource awareness, no adaptation (ablation)."""

    def __init__(self, first, second, theta, name=None, signal="margin"):
        self.v1, self.v2, self.theta, self.signal = first, second, theta, signal
        self.name = name or f"cascade_{first}>{second}@{signal}<{theta:.2f}"

    def first(self, ctx):
        return self.v1

    def second(self, ctx, r1):
        return self.v2 if signal_score(r1, self.signal) < self.theta else None


class ResourceAwareCascade(Policy):
    """STONE v2 router: resource-aware AND input-aware.

    1. Tier (from ResourceMonitor, in ctx['tier']) selects the ladder
       (first model, escalation model, latency budget) for that device state.
    2. The cheap model's uncertainty score (margin / confidence / entropy,
       chosen offline per tier) decides per image whether to escalate, with
       the threshold tuned online by a BudgetController per tier.
    3. 'critical' tier (hard thermal limit): stressed ladder's first model, never escalate.

    op_points (from bench/replay_eval.py -> configs/operating_points.json):
      {"idle":     {"first": "int8_160", "second": "fp32_224", "theta": 0.35,
                    "budget_ms": 90, "est_ms": {"int8_160": 30, ...}},
       "stressed": {...},
       "critical": {"first": "int8_96"}}            # optional
    """

    name = "v2_resource_aware"

    def __init__(self, op_points, eta=0.02, deadline_ms=None, adapt=True, name=None,
                 budget_mode="relative"):
        """budget_mode:
          'relative' (default): budget = ratio x (live EMA of the first-stage
             latency in this tier), ratio = budget_ms / est_ms[first] from the
             offline tuning. Transfers across regimes (back-to-back profiling
             vs. the protocol's 0.5 s gaps, idle vs. stressed), because all
             variants slow down roughly proportionally.
          'absolute': fixed budget_ms (a hard latency SLO)."""
        self.op = {k: dict(v) for k, v in op_points.items() if not k.startswith("_")}
        self.budget_mode = budget_mode
        if name:
            self.name = name
        if "critical" not in self.op:
            # no escalation, keep the stressed tier's (tuned) first model; the
            # tiniest variant costs too much accuracy for a modest saving
            base = self.op.get("stressed", self.op.get("idle"))
            self.op["critical"] = {"first": base["first"], "est_ms": base.get("est_ms", {})}
        self.ctrl = {}
        for t, o in self.op.items():
            if o.get("second"):
                lo, hi = SIGNALS[o.get("signal", "margin")]
                self.ctrl[t] = BudgetController(o["budget_ms"], o.get("theta", 0.3), eta=eta,
                                                theta_min=lo, theta_max=hi,
                                                deadline_ms=deadline_ms, adapt=adapt)
        self.tracker = LatencyTracker({t: o.get("est_ms", {}) for t, o in self.op.items()})
        self.ratio = {}
        for t, o in self.op.items():
            est_first = o.get("est_ms", {}).get(o["first"])
            if t in self.ctrl and est_first:
                self.ratio[t] = o["budget_ms"] / est_first

    def _sync_budget(self, t):
        if self.budget_mode == "relative" and t in self.ratio:
            cur = self.tracker.get(t, self.op[t]["first"])
            if cur:
                self.ctrl[t].budget_ms = self.ratio[t] * cur

    @classmethod
    def from_file(cls, path, **kw):
        with open(path) as f:
            return cls(json.load(f), **kw)

    def _tier(self, ctx):
        t = ctx.get("tier", "idle")
        return t if t in self.op else ("stressed" if t == "critical" else "idle")

    def first(self, ctx):
        return self.op[self._tier(ctx)]["first"]

    def second(self, ctx, r1):
        t = self._tier(ctx)
        o = self.op[t]
        if t == "critical" or t not in self.ctrl:
            return None
        c = self.ctrl[t]
        self._sync_budget(t)
        est1 = self.tracker.get(t, o["first"])
        est2 = self.tracker.get(t, o["second"])
        score = signal_score(r1, o.get("signal", "margin"))
        return o["second"] if c.should_escalate(score, est1, est2) else None

    def observe(self, ctx, results, total_ms):
        t = self._tier(ctx)
        for r in results:
            if r is not None:
                self.tracker.update(t, r["variant"], r["latency_ms"])
        if t in self.ctrl:
            self._sync_budget(t)
            self.ctrl[t].observe(total_ms, escalated=len([r for r in results if r]) > 1)

    def theta(self, ctx):
        t = self._tier(ctx)
        return self.ctrl[t].theta if t in self.ctrl else None

    def state(self):
        return {t: round(c.theta, 4) for t, c in self.ctrl.items()}

    def budgets(self):
        return {t: round(c.budget_ms, 3) for t, c in self.ctrl.items()}


def his_controller_policy(kind, state_path):
    """Wrap the ORIGINAL controllers (runtime/decision_engine.py, imported
    read-only) so they can be replayed / re-run beside v2. Returns an object
    with .decide(telemetry) -> 'fp32'|'int8' and .update(...)."""
    from runtime import decision_engine as de

    class _His:
        def __init__(self):
            self.name = kind
            name = {"linucb": "linucb", "eightsignal": "eightsignal", "egreedy": "egreedy"}[kind]
            kw = {"alpha": 1.0, "state_path": state_path}
            if name == "egreedy":
                kw = {"epsilon": 0.1, "state_path": state_path}
            self.algo = de.get_algo(name, **kw)
            self.de = de

        def decide(self, t):
            self.de.set_extra_telemetry(self.algo, t.get("cpu_freq_current", 0.0),
                                        t.get("cpu_freq_max") or 4000.0, t.get("disk_busy_time", 0))
            m, scores = self.de.choose(self.algo, t["cpu"], t["ram"], t["temperature"], t["battery"])
            return m, scores, self.de.get_last_source(self.algo) or "algo"

        def update(self, model, t, reward, confidence):
            self.de.update(self.algo, model, t["cpu"], t["ram"], t["temperature"], t["battery"],
                           reward, confidence=confidence)

    return _His()
