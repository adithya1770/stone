import numpy as np
import pytest
from PIL import Image

from v2.routing.budget import BudgetController, LatencyTracker
from v2.routing.frame_gate import FrameGate
from v2.routing.policies import FixedCascade, RandomMix, ResourceAwareCascade, Static
from v2.routing.resource_state import ResourceMonitor

OP = {
    "idle": {"first": "int8_160", "second": "fp32_224", "theta": 0.3, "budget_ms": 50.0,
             "est_ms": {"int8_160": 10.0, "fp32_224": 40.0, "int8_96": 4.0}},
    "stressed": {"first": "int8_128", "second": "int8_224", "theta": 0.3, "budget_ms": 50.0,
                 "est_ms": {"int8_128": 15.0, "int8_224": 45.0, "int8_96": 6.0}},
}


# ---------------- budget controller
def test_budget_theta_falls_when_over_budget_and_rises_when_under():
    c = BudgetController(100, theta0=0.5, eta=0.1)
    c.observe(200)
    assert c.theta < 0.5
    t = c.theta
    c.observe(10)
    assert c.theta > t


def test_budget_theta_clipped():
    c = BudgetController(10, theta0=0.5, eta=1.0)
    for _ in range(50):
        c.observe(1000)
    assert c.theta == 0.0
    for _ in range(500):
        c.observe(0)
    assert c.theta == 1.0


def test_budget_converges_to_budget_on_synthetic_stream():
    """Cheap model 20 ms, escalation +60 ms, margins uniform: the mean
    latency must settle near the 50 ms budget."""
    rng = np.random.default_rng(0)
    c = BudgetController(50, theta0=0.0, eta=0.02)
    lats = []
    for _ in range(5000):
        m = rng.random()
        esc = c.should_escalate(m)
        lat = 20 + (60 if esc else 0)
        c.observe(lat, esc)
        lats.append(lat)
    assert np.mean(lats[-2000:]) == pytest.approx(50, abs=3)


def test_deadline_guard_blocks_escalation():
    c = BudgetController(100, theta0=1.0, deadline_ms=50)
    assert not c.should_escalate(0.0, est_first_ms=20, est_second_ms=40)
    assert c.should_escalate(0.0, est_first_ms=20, est_second_ms=20)


def test_latency_tracker_ema():
    t = LatencyTracker({"idle": {"a": 10.0}}, alpha=0.5)
    t.update("idle", "a", 20.0)
    assert t.get("idle", "a") == 15.0


# ---------------- policies
def test_static_and_fixed_cascade():
    assert Static("int8_224").first({}) == "int8_224"
    p = FixedCascade("int8_128", "fp32_224", 0.4)
    assert p.second({}, {"margin": 0.1}) == "fp32_224"
    assert p.second({}, {"margin": 0.9}) is None


def test_random_mix_rate():
    p = RandomMix("int8_224", "fp32_224", 0.3, seed=1)
    picks = [p.first({}) for _ in range(20000)]
    assert np.mean([x == "fp32_224" for x in picks]) == pytest.approx(0.3, abs=0.02)


def test_resource_aware_uses_tier_ladder():
    p = ResourceAwareCascade(OP)
    assert p.first({"tier": "idle"}) == "int8_160"
    assert p.first({"tier": "stressed"}) == "int8_128"
    assert p.second({"tier": "idle"}, {"margin": 0.05, "variant": "int8_160"}) == "fp32_224"
    assert p.second({"tier": "stressed"}, {"margin": 0.05, "variant": "int8_128"}) == "int8_224"
    assert p.second({"tier": "idle"}, {"margin": 0.95, "variant": "int8_160"}) is None


def test_resource_aware_critical_uses_stressed_first_and_never_escalates():
    p = ResourceAwareCascade(OP)
    assert p.first({"tier": "critical"}) == "int8_128"
    assert p.second({"tier": "critical"}, {"margin": 0.0, "variant": "int8_128"}) is None


def test_resource_aware_adapts_per_tier_independently():
    p = ResourceAwareCascade(OP, eta=0.1)
    th_idle = p.ctrl["idle"].theta
    p.observe({"tier": "stressed"}, [{"variant": "int8_128", "latency_ms": 200.0}], 200.0)
    assert p.ctrl["stressed"].theta < 0.3
    assert p.ctrl["idle"].theta == th_idle


# ---------------- resource monitor
def test_monitor_tiers_and_hysteresis():
    m = ResourceMonitor(cpu_hi=60, dwell=3)
    calm = {"cpu_external": 10, "temperature": 50}
    hot = {"cpu_external": 95, "temperature": 50}
    assert m.update(calm) == "idle"
    assert m.update(hot) == "stressed"          # tighten immediately
    assert m.update(calm) == "stressed"         # relax only after dwell
    assert m.update(calm) == "stressed"
    assert m.update(calm) == "idle"


def test_monitor_critical_only_on_hard_limit_or_active_throttle():
    assert ResourceMonitor().update({"cpu_external": 0, "temperature": 82}) == "stressed"   # hot, not critical
    assert ResourceMonitor().update({"cpu_external": 0, "temperature": 86}) == "critical"
    assert ResourceMonitor().update({"cpu_external": 0, "temperature": 50, "throttled": 0x4}) == "critical"
    assert ResourceMonitor().update({"cpu_external": 0, "temperature": 50, "throttled": 0x1}) == "idle"  # under-voltage only


def test_monitor_falls_back_to_original_cpu_key():
    m = ResourceMonitor(cpu_hi=70)
    assert m.update({"cpu": 99.0, "temperature": 45}) == "stressed"


def test_monitor_eightsignal_mode_runs():
    m = ResourceMonitor(mode="eightsignal")
    for _ in range(5):
        tier = m.update({"cpu": 10, "ram": 30, "temperature": 45, "battery": 100})
    assert tier in ("idle", "stressed")
    assert m.update({"cpu": 99, "ram": 30, "temperature": 45, "battery": 100}) == "stressed"


# ---------------- frame gate
def test_frame_gate_reuses_identical_frames_only():
    g = FrameGate(threshold=6)
    a = Image.new("RGB", (64, 64), (100, 100, 100))
    b = Image.new("RGB", (64, 64), (200, 50, 10))
    cached, _, sig = g.check(a)
    assert cached is None
    g.store(sig, {"label": "x"})
    cached, diff, _ = g.check(a)
    assert cached == {"label": "x"} and diff == 0
    cached, diff, _ = g.check(b)
    assert cached is None and diff > 6


def test_signal_scores_and_entropy_bounds():
    from v2.routing.budget import SIGNALS, signal_score
    r = {"margin": 0.2, "confidence": 0.7, "entropy": 1.5}
    assert signal_score(r, "margin") == 0.2 and signal_score(r, "confidence") == 0.7
    assert signal_score(r, "entropy") == -1.5
    op = {"idle": {"first": "a_1", "second": "b_1", "theta": -2.0, "signal": "entropy", "budget_ms": 10, "est_ms": {}}}
    p = ResourceAwareCascade(op, eta=0.5)
    assert p.second({"tier": "idle"}, {"variant": "a_1", "margin": 0.9, "confidence": 0.9, "entropy": 3.0}) == "b_1"
    assert p.second({"tier": "idle"}, {"variant": "a_1", "margin": 0.0, "confidence": 0.1, "entropy": 0.5}) is None
    for _ in range(100):
        p.observe({"tier": "idle"}, [None], 1000.0)
    assert p.ctrl["idle"].theta == SIGNALS["entropy"][0]


def test_relative_budget_follows_first_stage_latency():
    op = {"idle": {"first": "a_1", "second": "b_1", "theta": 0.5, "budget_ms": 20.0,
                   "est_ms": {"a_1": 10.0, "b_1": 30.0}}}
    p = ResourceAwareCascade(op, budget_mode="relative")
    assert p.ratio["idle"] == pytest.approx(2.0)
    for _ in range(200):  # the device is now 3x slower for everything
        p.observe({"tier": "idle"}, [{"variant": "a_1", "latency_ms": 30.0}], 30.0)
    assert p.ctrl["idle"].budget_ms == pytest.approx(60.0, rel=0.02)
    q = ResourceAwareCascade(op, budget_mode="absolute")
    q.observe({"tier": "idle"}, [{"variant": "a_1", "latency_ms": 30.0}], 30.0)
    assert q.ctrl["idle"].budget_ms == 20.0


# ---------------- EdgeMLBalancer-style epsilon-greedy
def test_egreedy_ladder_tries_all_then_exploits_best_score():
    from v2.routing.bandits import EpsilonGreedyLadder
    p = EpsilonGreedyLadder(["a_1", "b_1", "c_1"], budget_ms=100, epsilon=0.0)
    ctx = {"tier": "idle"}
    conf = {"a_1": 0.5, "b_1": 0.9, "c_1": 0.95}
    lat = {"a_1": 50.0, "b_1": 90.0, "c_1": 300.0}      # c is over budget -> penalised
    for _ in range(3):
        v = p.first(ctx)
        p.observe(ctx, [{"variant": v, "confidence": conf[v]}, None], lat[v])
    assert {p.first(ctx) for _ in range(5)} == {"b_1"}
    assert p.score(0.95, 300.0) < p.score(0.9, 90.0)


def test_egreedy_ladder_explores_and_keeps_tiers_separate():
    from v2.routing.bandits import EpsilonGreedyLadder
    p = EpsilonGreedyLadder(["a_1", "b_1"], epsilon=0.5, seed=1)
    for t in ("idle", "stressed"):
        for v in ("a_1", "b_1"):
            p.observe({"tier": t}, [{"variant": v, "confidence": 0.9 if (v == "a_1") == (t == "idle") else 0.1}, None], 10.0)
    picks = [p.first({"tier": "idle"}) for _ in range(400)]
    assert 0.6 < picks.count("a_1") / 400 < 0.9          # mostly best, sometimes explores
    assert p.state()["stressed"]["b_1"] > p.state()["stressed"]["a_1"]
