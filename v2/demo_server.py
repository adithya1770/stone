"""STONE v2 live demo on the Raspberry Pi: the system we measured.

    python -m v2.demo_server                       # http://<pi>:8001  (v2 page)
    python -m v2.demo_server --certfile cert.pem --keyfile key.pem   # https (phone camera)

Same idea as the original demo (web_demo/server.py, untouched): a phone points
its camera at photos on a screen, frames go to the Pi, the Pi answers. Here the
Pi runs v2 on the model ladder:

  device layer   the group's EightSignal controller (unchanged) decides idle / stressed
                 (+ critical at >= 85 C or firmware throttling)
  photo layer    MobileNetV2 answers first; if it is unsure the photo goes to a bigger
                 EfficientNet-Lite model
  time budget    the confidence threshold adapts so the average time per photo stays at
                 --target-ms in every device state

The page shows, per frame: the answer, which model(s) ran and how long each took,
the device state, and a live chart of time per photo against the budget. A mode
switch compares v2 with 'fixed' (one model that fits the budget under stress) and
'big' (the best idle model, kept under stress) on the same camera feed.

Pages: /  (v2 demo page)   /classic  (the original page, read-only)
Needs configs/operating_points_ladder.json (written by v2/pi_run_ladder.sh).
"""
import argparse
import collections
import csv
import json
import os
import threading
import time
from datetime import datetime
from io import BytesIO

from PIL import Image

from v2.common.engine import MultiVariantEngine
from v2.common.paths import CONFIG_ROOT, REPO_ROOT, V2_ROOT, assert_inside_v2, enter_repo, results_path
from v2.routing.eightsignal_cascade import EightSignalCascade
from v2.routing.frame_gate import FrameGate
from v2.routing.policies import Static
from v2.routing.telemetry import BackgroundTelemetry

LADDER_OP = os.path.join(CONFIG_ROOT, "operating_points_ladder.json")
PAGE = os.path.join(V2_ROOT, "web", "index.html")
NICE = {"mnv2q": "MobileNetV2", "efl0q": "EfficientNet-Lite0", "efl1q": "EfficientNet-Lite1",
        "efl2q": "EfficientNet-Lite2", "efl3q": "EfficientNet-Lite3", "efl4q": "EfficientNet-Lite4",
        "int8": "MobileNetV2 (original INT8)", "fp32": "MobileNetV2 (original FP32)"}
FIELDS = ["request_id", "timestamp", "mode", "tier", "eightsignal", "cpu_external", "temperature", "throttled",
          "path", "escalated", "reused", "label", "confidence", "latency_ms", "e2e_ms", "true_label", "correct"]


def nice(variant):
    return NICE.get(variant.rsplit("_", 1)[0], variant)


def build_app(op_path=LADDER_OP, target_ms=145.0, reuse=False, log_path=None, state_path=None):
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    from runtime.reward import calculate_reward  # original, read-only

    with open(op_path) as f:
        op = json.load(f)
    for t, o in op.items():
        if not t.startswith("_"):
            o["budget_ms"] = float(target_ms)          # fixed time budget (ms) in every state
    ref = op["_reference"]
    v2 = EightSignalCascade(op, state_path or results_path("state", "demo_eightsignal.npz"),
                            name="v2", budget_mode="absolute")
    modes = {"v2": v2.cascade,
             "fixed": Static(ref["fixed_model"], "fixed"),
             "big": Static(ref["best_static_per_state"]["idle"], "big")}
    variants = sorted({o[k] for t, o in op.items() if not t.startswith("_") for k in ("first", "second") if o.get(k)}
                      | {ref["fixed_model"], ref["best_static_per_state"]["idle"]})
    engine = MultiVariantEngine(variants)
    tel = BackgroundTelemetry().start()
    gate = FrameGate(enabled=True)
    lock = threading.Lock()
    st = {"mode": "v2", "reuse": bool(reuse), "n": 0}
    recent = collections.deque(maxlen=30)      # {"ms", "escalated", "mode"} of computed (not reused) frames
    log_path = log_path or assert_inside_v2(results_path("demo_log.csv"))
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)

    app = FastAPI()

    def summary():
        same = [r for r in recent if r["mode"] == st["mode"]]
        return {"avg_ms": round(sum(r["ms"] for r in same) / len(same), 1) if same else None,
                "escalated_pct": round(100 * sum(r["escalated"] for r in same) / len(same), 1) if same else None,
                "window": len(same)}

    @app.get("/")
    def home():
        return FileResponse(PAGE)

    @app.get("/classic")
    def classic():
        return FileResponse(os.path.join(REPO_ROOT, "web_demo", "static", "index.html"))

    @app.get("/health")
    def health():
        return {"status": "ok", "device": "raspberry-pi", "version": "stone-v2-demo"}

    @app.get("/stats")
    def stats():
        return {"mode": st["mode"], "reuse": st["reuse"], "requests": st["n"], "target_ms": target_ms,
                "models": {m: (nice(p.variant) if hasattr(p, "variant") else
                               {t: [nice(o["first"])] + ([nice(o["second"])] if o.get("second") else [])
                                for t, o in op.items() if not t.startswith("_")})
                           for m, p in modes.items()},
                "theta": v2.state(), "telemetry": tel.snapshot(), **summary()}

    class Settings(BaseModel):
        mode: str | None = None
        reuse: bool | None = None

    @app.post("/settings")
    def settings(s: Settings):
        with lock:
            if s.mode in modes:
                st["mode"] = s.mode
            if s.reuse is not None:
                st["reuse"] = s.reuse
        return {"mode": st["mode"], "reuse": st["reuse"]}

    @app.post("/infer")
    async def infer(file: UploadFile = File(...)):
        data = await file.read()
        with lock:  # one inference at a time, like the original
            w0 = time.perf_counter()
            img = Image.open(BytesIO(data)).convert("RGB")
            t = tel.snapshot()
            # EightSignal sees the load from OTHER processes: the background reader's
            # cpu_external excludes this server's own inference work.
            t_dev = dict(t, cpu=t.get("cpu_external", t["cpu"]))
            d0 = time.perf_counter()
            tier = v2.device_tier(t_dev)
            pol = modes[st["mode"]]
            ctx = {"tier": tier}
            cached, diff, sig = gate.check(img) if st["reuse"] else (None, None, None)
            reused = cached is not None
            if reused:
                path, fin, total, escalated = cached["path"], cached["final"], 0.0, False
                dms = (time.perf_counter() - d0) * 1000
            else:
                v1 = pol.first(ctx)
                dms = (time.perf_counter() - d0) * 1000
                r1 = engine.run_image(img, v1)
                vsecond = pol.second(ctx, r1)
                r2 = engine.run_image(img, vsecond) if vsecond else None
                total = r1["latency_ms"] + (r2["latency_ms"] if r2 else 0.0)
                pol.observe(ctx, [r1, r2], total)
                fin = r2 or r1
                escalated = r2 is not None
                path = [{"model": nice(r["variant"]), "variant": r["variant"], "ms": round(r["latency_ms"], 1),
                         "label": r["label"], "confidence": r["confidence"]} for r in (r1, r2) if r]
                v2.feedback(t_dev, calculate_reward(confidence=fin["confidence"], latency_ms=total, cpu=t_dev["cpu"],
                                                    ram=t["ram"], temperature=t["temperature"], decision_time_ms=dms),
                            fin["confidence"])
                recent.append({"ms": total, "escalated": escalated, "mode": st["mode"]})
                if st["reuse"]:
                    gate.store(sig, {"path": path, "final": fin})
            e2e = (time.perf_counter() - w0) * 1000
            st["n"] += 1
            rid = st["n"]
            es = v2.last_decision[0] if v2.last_decision else ""
            row = {"request_id": rid, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "mode": st["mode"], "tier": tier, "eightsignal": es, "cpu_external": t.get("cpu_external"),
                   "temperature": t["temperature"], "throttled": t["throttled"],
                   "path": " > ".join(f"{p['variant']}:{p['ms']}" for p in path), "escalated": escalated,
                   "reused": reused, "label": fin["label"], "confidence": fin["confidence"],
                   "latency_ms": round(total, 2), "e2e_ms": round(e2e, 2), "true_label": "", "correct": ""}
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            th = v2.theta(ctx) if st["mode"] == "v2" else None
            out = {"request_id": rid, "label": fin["label"], "confidence": fin["confidence"],
                   "model": path[-1]["model"], "path": path, "escalated": escalated, "reused": reused,
                   "latency_ms": round(total, 1), "e2e_ms": round(e2e, 1), "decision_time_ms": round(dms, 2),
                   "mode": st["mode"], "tier": tier, "eightsignal": es, "target_ms": target_ms,
                   "theta": round(th, 3) if isinstance(th, float) else None,
                   "cpu": t.get("cpu_external"), "ram": t["ram"], "temperature": t["temperature"],
                   "throttled": t["throttled"], **summary()}
        return out

    class LabelPayload(BaseModel):
        request_id: int
        true_label: str

    @app.post("/label")
    def label(payload: LabelPayload):
        with lock:
            with open(log_path, newline="") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                if r["request_id"] == str(payload.request_id):
                    r["true_label"] = payload.true_label
                    r["correct"] = str(r["label"].strip().lower() == payload.true_label.strip().lower())
            with open(log_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
        return {"status": "ok"}

    app.state.telemetry, app.state.v2, app.state.modes = tel, v2, modes
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--op-points", default=LADDER_OP)
    ap.add_argument("--target-ms", type=float, default=145.0,
                    help="average time per photo to aim for, in every device state (145 lands at about 150 on the Pi)")
    ap.add_argument("--reuse", action="store_true", help="reuse the last answer for near-identical frames")
    ap.add_argument("--certfile", default=None, help="HTTPS certificate (phones only allow the camera over https)")
    ap.add_argument("--keyfile", default=None)
    a = ap.parse_args()
    enter_repo()
    if not os.path.exists(a.op_points):
        raise SystemExit(f"missing {a.op_points}: run v2/pi_run_ladder.sh first")
    import uvicorn
    print(f"STONE v2 demo: target {a.target_ms:.0f} ms per photo, operating points {a.op_points}")
    uvicorn.run(build_app(a.op_points, a.target_ms, a.reuse), host=a.host, port=a.port,
                ssl_certfile=a.certfile, ssl_keyfile=a.keyfile)


if __name__ == "__main__":
    main()
