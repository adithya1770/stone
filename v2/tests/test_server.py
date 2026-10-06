"""Smoke test of server_v2 with FastAPI's TestClient (no network)."""
import io
import os

import pytest
from PIL import Image


def test_server_infer_label_stats(images, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from v2.server_v2 import build_app

    app = build_app(str(tmp_path / "missing.json"), reuse=True, log_path=str(tmp_path / "log.csv"))  # falls back to defaults
    c = TestClient(app)
    assert c.get("/health").json()["version"] == "stone-v2"
    buf = io.BytesIO()
    Image.open(images[0][1]).convert("RGB").save(buf, "JPEG")
    r1 = c.post("/infer", files={"file": ("f.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert r1["model"].startswith(("int8_", "fp32_")) and not r1["reused"]
    r2 = c.post("/infer", files={"file": ("f.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert r2["reused"] and r2["label"] == r1["label"]       # identical frame -> reused
    assert c.post("/label", json={"request_id": r1["request_id"], "true_label": r1["label"]}).json()["status"] == "ok"
    s = c.get("/stats").json()
    assert s["requests"] == 2 and s["reused_pct"] == 50
    app.state.telemetry.stop()
