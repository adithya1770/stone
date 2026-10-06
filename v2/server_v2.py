"""STONE v2 live server (the original web_demo/server.py is untouched).

    python -m v2.server_v2                 # http://<pi>:8001  (original uses 8000)

Default system: v2_final (the group's EightSignal controller decides the device
state, the per-photo check decides escalation, rebuilt INT8/FP32 models) when
configs/operating_points_final.json exists; otherwise the earlier v2 router.
    python -m v2.server_v2 --system v2     # earlier router (operating_points.json)

Same endpoints and page as the original demo (/, /infer, /label, /health),
plus /stats. Differences:
  - telemetry comes from a background thread: no 0.5 s wait per request
  - routing = ResourceAwareCascade (tier -> ladder, margin -> escalate,
    latency budget adapted online; no labels needed)
  - optional frame reuse for near-identical consecutive camera frames
  - logs to v2/results_v2/web_demo_v2_log.csv with e2e time per request
"""
import argparse
import csv
import os
import threading
import time
from datetime import datetime
from io import BytesIO

from PIL import Image

from v2.common.paths import CONFIG_ROOT, REPO_ROOT, assert_inside_v2, enter_repo, results_path
from v2.common.engine import MultiVariantEngine
from v2.routing.frame_gate import FrameGate
from v2.routing.eightsignal_cascade import EightSignalCascade
from v2.routing.policies import ResourceAwareCascade
from v2.routing.resource_state import ResourceMonitor
from v2.routing.telemetry import BackgroundTelemetry

FIELDS = ["request_id", "timestamp", "cpu", "cpu_external", "ram", "temperature", "throttled", "tier",
          "model", "first_variant", "escalated", "reused", "frame_diff", "theta", "true_label", "label",
          "correct", "confidence", "margin", "latency_ms", "e2e_ms", "decision_time_ms"]

DEFAULT_OP = {  # used only if configs/operating_points.json does not exist yet
    "idle": {"first": "int8_160", "second": "fp32_224", "theta": 0.3, "budget_ms": 100.0, "est_ms": {}},
    "stressed": {"first": "int8_128", "second": "int8_224", "theta": 0.2, "budget_ms": 100.0, "est_ms": {}},
}


FINAL_OP = os.path.join(CONFIG_ROOT, "operating_points_final.json")


def build_app(op_path, reuse=True, budget_scale=1.0, monitor_mode="simple", log_path=None,
              budget_mode="relative", system="v2"):
    """system='final': EightSignalCascade (EightSignal device layer + photo check);
    system='v2': ResourceAwareCascade with ResourceMonitor(monitor_mode)."""
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    if system == "final":
        import json
        with open(op_path) as f:
            final = EightSignalCascade(json.load(f), results_path("state", "server_v2_final_eightsignal.npz"),
                                       name="v2_final", budget_mode=budget_mode)
        policy = final.cascade   # the photo layer (budgets, thresholds, stats)
    elif os.path.exists(op_path):
        final = None
        policy = ResourceAwareCascade.from_file(op_path, budget_mode=budget_mode)
    else:
        final = None
        print(f"WARNING: {op_path} not found, using built-in defaults")
        policy = ResourceAwareCascade(DEFAULT_OP, budget_mode="absolute")
    for t in policy.ctrl:
        policy.ctrl[t].budget_ms *= budget_scale
        if t in policy.ratio:
            policy.ratio[t] *= budget_scale
    variants = sorted({o[k] for o in policy.op.values() for k in ("first", "second") if o.get(k)})
    engine = MultiVariantEngine(variants)
    tel = BackgroundTelemetry().start()
    monitor = ResourceMonitor(mode=monitor_mode)
    tier_now = {"tier": "idle"}
    gate = FrameGate(enabled=reuse)
    lock = threading.Lock()
    log_path = log_path or assert_inside_v2(results_path("web_demo_v2_log.csv"))
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)
    counter = {"n": 0, "lat": [], "e2e": [], "esc": 0, "reused": 0}

    app = FastAPI()
    static_page = os.path.join(REPO_ROOT, "web_demo", "static", "index.html")  # original page, read-only

    @app.get("/")
    def home():
        return FileResponse(static_page)

    @app.get("/health")
    def health():
        return {"status": "ok", "device": "raspberry-pi", "version": "stone-v2"}

    @app.get("/stats")
    def stats():
        n = max(counter["n"], 1)
        return {"requests": counter["n"], "escalated_pct": 100 * counter["esc"] / n,
                "reused_pct": 100 * counter["reused"] / n, "theta": policy.state(),
                "budget_ms": policy.budgets(), "budget_mode": policy.budget_mode,
                "mean_latency_ms": sum(counter["lat"][-200:]) / max(len(counter["lat"][-200:]), 1),
                "mean_e2e_ms": sum(counter["e2e"][-200:]) / max(len(counter["e2e"][-200:]), 1),
                "tier": tier_now["tier"], "system": system, "telemetry": tel.snapshot(),
                "rss_mb": round(tel.rss_mb(), 1)}

    @app.post("/infer")
    async def infer(file: UploadFile = File(...)):
        data = await file.read()
        with lock:  # one inference at a time, like the original
            w0 = time.perf_counter()
            img = Image.open(BytesIO(data)).convert("RGB")
            t = tel.snapshot()
            d0 = time.perf_counter()
            tier = final.device_tier(t) if final else monitor.update(t)
            tier_now["tier"] = tier
            ctx = {"tier": tier}
            cached, diff, sig = gate.check(img)
            reused = cached is not None
            r2 = None
            if reused:
                r1 = dict(cached)
                total = 0.0
                dms = (time.perf_counter() - d0) * 1000
            else:
                v1 = policy.first(ctx)
                dms = (time.perf_counter() - d0) * 1000
                r1 = engine.run_image(img, v1)
                d1 = time.perf_counter()
                v2 = policy.second(ctx, r1)
                dms += (time.perf_counter() - d1) * 1000
                if v2:
                    r2 = engine.run_image(img, v2)
                total = r1["latency_ms"] + (r2["latency_ms"] if r2 else 0.0)
                policy.observe(ctx, [r1, r2], total)
            fin = r2 or r1
            if final is not None and not reused:   # original EightSignal update (same reward as original)
                from runtime.reward import calculate_reward  # original, read-only
                final.feedback(t, calculate_reward(confidence=fin["confidence"], latency_ms=total, cpu=t["cpu"],
                                                   ram=t["ram"], temperature=t["temperature"], decision_time_ms=dms),
                               fin["confidence"])
            final_r = fin
            if not reused:
                gate.store(sig, final_r)
            e2e = (time.perf_counter() - w0) * 1000
            counter["n"] += 1
            counter["lat"].append(total)
            counter["e2e"].append(e2e)
            counter["esc"] += int(r2 is not None)
            counter["reused"] += int(reused)
            rid = counter["n"]
            th = policy.theta(ctx)
            row = {"request_id": rid, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "cpu": t["cpu"], "cpu_external": t["cpu_external"], "ram": t["ram"],
                   "temperature": t["temperature"], "throttled": t["throttled"], "tier": tier,
                   "model": final_r["variant"], "first_variant": r1["variant"], "escalated": r2 is not None,
                   "reused": reused, "frame_diff": round(diff, 2) if diff is not None else "",
                   "theta": round(th, 4) if th is not None else "", "true_label": "", "label": final_r["label"],
                   "correct": "", "confidence": final_r["confidence"], "margin": final_r["margin"],
                   "latency_ms": round(total, 3), "e2e_ms": round(e2e, 3), "decision_time_ms": round(dms, 3)}
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        return {"request_id": rid, "model": final_r["variant"], "label": final_r["label"],
                "confidence": final_r["confidence"], "latency_ms": round(total, 2),
                "decision_time_ms": round(dms, 3),
                "decision_source": f"{'v2_final' if final else 'v2'}:{tier}" + (":reused" if reused else ""),
                "cpu": t["cpu"], "ram": t["ram"], "temperature": t["temperature"],
                "escalated": r2 is not None, "reused": reused, "e2e_ms": round(e2e, 2)}

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

    app.state.policy, app.state.telemetry, app.state.gate = policy, tel, gate
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--system", default=None, choices=["final", "v2"],
                    help="default: final if configs/operating_points_final.json exists, else v2")
    ap.add_argument("--op-points", default=None,
                    help="default: operating_points_final.json (final) / operating_points.json (v2)")
    ap.add_argument("--no-reuse", action="store_true", help="disable frame reuse")
    ap.add_argument("--monitor", default="simple", choices=["simple", "eightsignal"])
    ap.add_argument("--budget-mode", default="relative", choices=["relative", "absolute"])
    ap.add_argument("--budget-scale", type=float, default=1.0, help="e.g. 0.8 = 20%% tighter latency budget")
    a = ap.parse_args()
    enter_repo()
    system = a.system or ("final" if os.path.exists(FINAL_OP) else "v2")
    op_path = a.op_points or (FINAL_OP if system == "final" else os.path.join(CONFIG_ROOT, "operating_points.json"))
    print(f"STONE v2 server: system={system}, operating points={op_path}")
    import uvicorn
    uvicorn.run(build_app(op_path, reuse=not a.no_reuse, monitor_mode=a.monitor, system=system,
                          budget_mode=a.budget_mode, budget_scale=a.budget_scale), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
