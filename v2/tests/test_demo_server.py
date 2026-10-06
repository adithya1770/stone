"""Smoke test of the live demo server (FastAPI TestClient, no network)."""
import io
import json

import pytest
from PIL import Image

OP = {
    "idle": {"first": "int8_160", "second": "fp32_224", "theta": 0.9, "signal": "confidence", "budget_ms": 150.0,
             "est_ms": {"int8_160": 10.0, "fp32_224": 40.0}},
    "stressed": {"first": "int8_160", "second": "int8_224", "theta": 0.2, "signal": "margin", "budget_ms": 150.0,
                 "est_ms": {"int8_160": 15.0, "int8_224": 30.0}},
    "_reference": {"best_static_per_state": {"idle": "fp32_224", "stressed": "int8_160"}, "fixed_model": "int8_160"},
}


def test_demo_server_modes_and_path(images, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from v2.demo_server import build_app
    op_path = tmp_path / "op.json"
    op_path.write_text(json.dumps(OP))
    app = build_app(str(op_path), target_ms=145, log_path=str(tmp_path / "log.csv"),
                    state_path=str(tmp_path / "es.npz"))
    c = TestClient(app)
    assert c.get("/health").json()["version"] == "stone-v2-demo"
    assert "STONE v2" in c.get("/").text
    buf = io.BytesIO()
    Image.open(images[0][1]).convert("RGB").save(buf, "JPEG")
    jpg = buf.getvalue()
    r = c.post("/infer", files={"file": ("f.jpg", jpg, "image/jpeg")}).json()
    assert r["mode"] == "v2" and r["target_ms"] == 145 and r["tier"] in ("idle", "stressed", "critical")
    assert 1 <= len(r["path"]) <= 2 and r["path"][0]["variant"] == "int8_160"
    assert r["escalated"] == (len(r["path"]) == 2)
    assert c.post("/settings", json={"mode": "big"}).json()["mode"] == "big"
    r2 = c.post("/infer", files={"file": ("f.jpg", jpg, "image/jpeg")}).json()
    assert r2["mode"] == "big" and [p["variant"] for p in r2["path"]] == ["fp32_224"]
    assert c.post("/settings", json={"mode": "nonsense"}).json()["mode"] == "big"
    c.post("/settings", json={"mode": "fixed", "reuse": True})
    a = c.post("/infer", files={"file": ("f.jpg", jpg, "image/jpeg")}).json()
    b = c.post("/infer", files={"file": ("f.jpg", jpg, "image/jpeg")}).json()
    assert not a["reused"] and b["reused"] and b["label"] == a["label"]
    assert c.post("/label", json={"request_id": a["request_id"], "true_label": a["label"]}).json()["status"] == "ok"
    s = c.get("/stats").json()
    assert s["requests"] == 4 and s["mode"] == "fixed"
    app.state.telemetry.stop()
