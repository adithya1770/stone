"""v2_final: EightSignal device layer + per-photo check."""
import io
import json
import os

import pytest
from PIL import Image

OP = {
    "idle": {"first": "int8_160", "second": "fp32_224", "theta": 0.5, "signal": "confidence", "budget_ms": 50.0,
             "est_ms": {"int8_160": 10.0, "fp32_224": 40.0}},
    "stressed": {"first": "int8_224", "second": None, "theta": 0.0, "budget_ms": 30.0,
                 "est_ms": {"int8_224": 30.0}},
}
CALM = {"cpu": 5.0, "ram": 30.0, "temperature": 45.0, "battery": 100.0,
        "cpu_freq_current": 1800.0, "cpu_freq_max": 1800.0}
BUSY = dict(CALM, cpu=100.0, temperature=80.0, cpu_freq_current=900.0, ram=90.0)  # several bad signals


def _final(tmp_path):
    from v2.routing.eightsignal_cascade import EightSignalCascade
    return EightSignalCascade(OP, str(tmp_path / "es.npz"))


def test_tier_follows_eightsignal_decision(tmp_path):
    p = _final(tmp_path)
    for _ in range(10):
        assert p.device_tier(CALM) == "idle"
    tiers = [p.device_tier(BUSY) for _ in range(5)]
    assert tiers == ["stressed"] * 5 and p.last_decision[0] == "int8"   # EightSignal votes int8
    assert p.first({"tier": "stressed"}) == "int8_224"
    assert p.second({"tier": "stressed"}, {"variant": "int8_224", "confidence": 0.0, "margin": 0.0}) is None


def test_idle_tier_escalates_only_unsure_photos(tmp_path):
    p = _final(tmp_path)
    ctx = {"tier": "idle"}
    assert p.first(ctx) == "int8_160"
    assert p.second(ctx, {"variant": "int8_160", "confidence": 0.2, "margin": 0.1}) == "fp32_224"
    assert p.second(ctx, {"variant": "int8_160", "confidence": 0.95, "margin": 0.9}) is None


def test_critical_overrides_and_never_escalates(tmp_path):
    p = _final(tmp_path)
    assert p.device_tier(dict(CALM, temperature=86.0)) == "critical"
    assert p.device_tier(dict(CALM, throttled=0x4)) == "critical"
    assert p.second({"tier": "critical"}, {"variant": "int8_224", "confidence": 0.0, "margin": 0.0}) is None


def test_feedback_reaches_original_controller(tmp_path):
    p = _final(tmp_path)
    p.device_tier(CALM)
    p.feedback(CALM, reward=0.5, confidence=0.3)
    assert p.his.algo["state"]["prev_confidence"] == pytest.approx(0.3)


def test_server_runs_final_system(images, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from v2.server_v2 import build_app
    op_path = tmp_path / "op.json"
    op_path.write_text(json.dumps(OP))
    app = build_app(str(op_path), reuse=False, log_path=str(tmp_path / "log.csv"), system="final")
    c = TestClient(app)
    buf = io.BytesIO()
    Image.open(images[0][1]).convert("RGB").save(buf, "JPEG")
    r = c.post("/infer", files={"file": ("f.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert r["decision_source"].startswith("v2_final:")
    assert r["model"] in ("int8_160", "fp32_224", "int8_224")
    assert c.get("/stats").json()["system"] == "final"
    app.state.telemetry.stop()
