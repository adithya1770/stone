"""STONE v2 final router: the group's EightSignal controller + a per-photo check.

Two layers, each answering one question:

  1. Device layer - "can the Pi afford the big model right now?"
     The ORIGINAL EightSignal controller (runtime/decision_engine.py, imported
     read-only, unchanged: 8 votes, LinUCB in the grey zone, hysteresis,
     reward/confidence updates). Its decision is used as the device state:
        EightSignal says 'fp32'  -> tier 'idle'      (headroom available)
        EightSignal says 'int8'  -> tier 'stressed'  (stay cheap)
     plus one safety rule on top: >= 85 C or firmware 'currently throttled'
     -> tier 'critical' (cheapest model, never escalate).

  2. Photo layer - "does THIS photo need the big model?"
     ResourceAwareCascade: the tier's cheap model answers first; if its
     confidence (margin / confidence / entropy, chosen offline) is below a
     threshold the photo is escalated to the tier's stronger model. The
     threshold adapts online to keep each tier inside its time budget.

With the photo layer switched off this is exactly EightSignal choosing
between the two models (config 'eightsignal_rebuilt'), so the comparison
v2_final vs eightsignal_rebuilt isolates what the photo check adds.
"""
from .policies import ResourceAwareCascade, his_controller_policy
from .resource_state import ResourceMonitor

TIER_OF = {"fp32": "idle", "int8": "stressed"}


class EightSignalCascade:
    name = "v2_final"

    def __init__(self, op_points, state_path, name=None, temp_critical=85.0, **cascade_kw):
        self.his = his_controller_policy("eightsignal", state_path)
        self.cascade = ResourceAwareCascade(op_points, **cascade_kw)
        self.temp_critical = temp_critical
        self.name = name or self.name
        self.cascade.name = self.name
        self.last_decision = None

    # ---- device layer --------------------------------------------------
    def device_tier(self, t):
        """Run the original EightSignal decision on this telemetry snapshot
        and map it to a tier. Called once per request, before inference."""
        m, scores, source = self.his.decide(t)
        self.last_decision = (m, source)
        if ResourceMonitor._hard_throttled(t) or t.get("temperature", 40.0) >= self.temp_critical:
            return "critical"
        return TIER_OF.get(m, "stressed")

    def feedback(self, t, reward, confidence):
        """Original EightSignal update (difficulty vote + LinUCB grey zone)."""
        if self.last_decision is not None:
            self.his.update(self.last_decision[0], t, reward, confidence)

    # ---- photo layer (delegated) ---------------------------------------
    def first(self, ctx):
        return self.cascade.first(ctx)

    def second(self, ctx, r1):
        return self.cascade.second(ctx, r1)

    def observe(self, ctx, results, total_ms):
        self.cascade.observe(ctx, results, total_ms)

    def theta(self, ctx):
        return self.cascade.theta(ctx)

    def state(self):
        return self.cascade.state()

    def budgets(self):
        return self.cascade.budgets()
