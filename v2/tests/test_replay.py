"""Replay maths on a synthetic response table with known answers."""
import csv

import numpy as np
import pytest

from v2.bench import replay_eval as R
from v2.routing.policies import FixedCascade, ResourceAwareCascade


@pytest.fixture
def table(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        hard = rng.random() < 0.3
        m_cheap = rng.uniform(0, 0.3) if hard else rng.uniform(0.4, 1.0)
        for v, ok, m, lat in [("int8_128", not hard, m_cheap, 10.0),
                              ("int8_224", not hard or rng.random() < 0.5, 0.5, 30.0),
                              ("fp32_224", True if hard else rng.random() < 0.97, 0.6, 50.0)]:
            rows.append({"image_id": f"img{i}", "true_index": 0, "variant": v, "pred_index": 0,
                         "correct": ok, "confidence": 0.5, "margin": m, "entropy": 1.0,
                         "latency_ms": lat, "preprocess_ms": 1.0})
    p = tmp_path / "t.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return R.load_table(str(p))


def test_cascade_extremes_equal_statics(table):
    L, _ = R.latency_matrix(table, None, 0)
    idx = np.arange(len(table["ids"]))
    s1 = R.static_eval(table, L, "int8_128", idx)
    s2 = R.static_eval(table, L, "fp32_224", idx)
    never = R.cascade_eval(table, L, "int8_128", "fp32_224", 0.0, idx)
    always = R.cascade_eval(table, L, "int8_128", "fp32_224", 1.01, idx)
    assert never["acc"] == pytest.approx(s1["acc"]) and never["lat"] == pytest.approx(10)
    assert always["acc"] == pytest.approx(s2["acc"]) and always["lat"] == pytest.approx(60)


def test_margin_cascade_beats_coin_flip_when_margin_is_informative(table):
    L, _ = R.latency_matrix(table, None, 0)
    idx = np.arange(len(table["ids"]))
    c = R.cascade_eval(table, L, "int8_128", "fp32_224", 0.35, idx)
    st = {v: R.static_eval(table, L, v, idx) for v in table["variants"]}
    st["int8_224"], st["fp32_224"] = st["int8_224"], st["fp32_224"]
    mix = R.mix_line_acc(st, c["lat"])
    assert c["acc"] > (mix if mix is not None else st["int8_224"]["acc"])


def test_oracle_upper_bounds_every_static(table):
    L, _ = R.latency_matrix(table, None, 0)
    idx = np.arange(len(table["ids"]))
    o = R.oracle_eval(table, L, idx)
    for v in table["variants"]:
        assert o["acc"] >= R.static_eval(table, L, v, idx)["acc"] - 1e-9


def test_select_respects_budget(table):
    L, _ = R.latency_matrix(table, None, 0)
    idx = np.arange(len(table["ids"]))
    ch = R.select(R.candidates(table, L, idx), budget=30.0)
    assert ch["lat"] <= 30.0 + 1e-9


def test_simulated_live_policy_matches_vectorised_cascade(table):
    """The live Policy class and the vectorised replay must agree when the
    budget controller is frozen (adapt=False)."""
    L, _ = R.latency_matrix(table, None, 0)
    idx = np.arange(len(table["ids"]))
    op = {"idle": {"first": "int8_128", "second": "fp32_224", "theta": 0.35, "budget_ms": 40.0, "est_ms": {}}}
    live, _, _ = R.simulate_policy(table, {"idle": L}, ResourceAwareCascade(op, adapt=False), idx, ["idle"] * len(idx))
    vec = R.cascade_eval(table, L, "int8_128", "fp32_224", 0.35, idx)
    assert live["acc"] == pytest.approx(vec["acc"]) and live["lat"] == pytest.approx(vec["lat"])


def test_split_is_deterministic():
    from v2.common.datasets import split_of
    assert split_of("abc") == split_of("abc")
    s = [split_of(f"x{i}") for i in range(2000)]
    assert 0.45 < s.count("tune") / len(s) < 0.55


def test_auroc_known_values():
    assert R.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert R.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert R.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == pytest.approx(0.5)


def test_kind_of_variant_names():
    for v, k in [("int8_224", "int8"), ("fp32n_160", "fp32n"), ("mnv2q_128", "mnv2q"), ("mnv3sq_224", "mnv3sq")]:
        assert v.rsplit("_", 1)[0] == k


def test_candidates_restricted_to_allowed_kinds(table):
    """--kinds: v2 may only choose among the allowed variants; the reference
    statics stay in the table for the budget and the paired tests."""
    L, _ = R.latency_matrix(table, None, 0)
    idx = np.arange(len(table["ids"]))
    table["cands"] = ["int8_128", "int8_224"]
    used = {v for c in R.candidates(table, L, idx) for v in (c.get("first"), c.get("second"), c.get("variant")) if v}
    assert used <= {"int8_128", "int8_224"}
    o = R.oracle_eval(table, L, idx)
    assert o["lat"] <= 30.0 + 1e-9


def test_replay_main_with_kinds_keeps_reference_rows(table, tmp_path, monkeypatch):
    import json
    import shutil
    import sys
    from v2.common.paths import RESULTS_ROOT
    src = tmp_path / "t2.csv"
    rows = []
    for v in table["variants"]:
        for k, i in enumerate(table["ids"]):
            rows.append({"image_id": i, "true_index": 0, "variant": v.replace("int8_128", "mnv2q_128"),
                         "pred_index": 0, "correct": bool(table["correct"][v][k]), "confidence": 0.5,
                         "margin": float(table["margin"][v][k]), "entropy": 1.0,
                         "latency_ms": {"int8_128": 10.0, "int8_224": 30.0, "fp32_224": 50.0}[v], "preprocess_ms": 1.0})
    with open(src, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = f"{RESULTS_ROOT}/_pytest_replay_kinds"
    try:
        monkeypatch.setattr(sys, "argv", ["replay_eval", "--table", str(src), "--kinds", "mnv2q", "--out-dir", out])
        R.main()
        s = json.load(open(f"{out}/summary.json"))
        assert s["v2_candidates"] == ["mnv2q_128"]
        assert {"int8_224", "fp32_224"} <= set(s["variants"])
        ch = s["conditions"]["idle"]["chosen_on_tune"]
        assert ch["first"] == "mnv2q_128" and ch.get("second") is None
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_select_acc_tol_prefers_faster_near_best():
    cands = [{"acc": 80.0, "lat": 70.0}, {"acc": 79.8, "lat": 50.0}, {"acc": 70.0, "lat": 10.0}]
    assert R.select(cands, budget=100.0)["lat"] == 70.0
    assert R.select(cands, budget=100.0, acc_tol=0.5)["lat"] == 50.0
    assert R.select(cands, budget=60.0, acc_tol=0.5)["lat"] == 50.0
