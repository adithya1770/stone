import numpy as np
import pytest

from v2.common.engine import MultiVariantEngine, load_rgb, parse_variant, scores_from_output
from v2.common.labels import load_labels, synsets, wnid_to_index


def test_synsets_and_imagenette_mapping_match_original():
    from runtime.label_lookup import WNID_TO_NAME, build_wnid_to_label_index  # original
    assert len(synsets()) == 1000
    labels = load_labels()
    his_map, _ = build_wnid_to_label_index("models/labels.txt")
    for wnid in WNID_TO_NAME:
        # original: label index into labels.txt (with 'background' at 0)
        assert wnid_to_index(wnid) + 1 == his_map[wnid], wnid
        assert labels[wnid_to_index(wnid) + 1].lower() == WNID_TO_NAME[wnid].lower()


def test_parse_variant():
    assert parse_variant("int8_128") == ("int8", 128)
    assert parse_variant("v3s_int8_224") == ("v3s_int8", 224)


def test_scores_from_output_margin_entropy():
    p = np.zeros(1000, np.float32)
    p[3], p[7], p[9] = 0.6, 0.3, 0.1
    _, conf, margin, ent, t1, t2 = scores_from_output(p)
    assert (t1, t2) == (3, 7)
    assert conf == pytest.approx(0.6) and margin == pytest.approx(0.3)
    assert ent > 0


def test_224_variants_bit_identical_to_original_engine(images):
    """v2's fp32_224 / int8_224 must reproduce runtime/inference_engine.py
    exactly, so v2 numbers are comparable with the original results."""
    from runtime.inference_engine import InferenceEngine  # original, read-only
    his = InferenceEngine("models/mobilenet_v2_fp32.tflite", "models/mobilenet_v2_int8.tflite", "models/labels.txt")
    eng = MultiVariantEngine(["fp32_224", "int8_224"])
    for _, path, _ in images:
        for kind in ("fp32", "int8"):
            a = his.run(path, kind)
            b = eng.run(path, f"{kind}_224")
            assert a["label"] == b["label"], (path, kind)
            assert a["confidence"] == pytest.approx(b["confidence"], abs=1e-4), (path, kind)


def test_resolution_ladder_runs_and_is_consistent(images):
    vs = ["int8_96", "int8_128", "int8_160", "fp32_128", "fp32n_224"]
    eng = MultiVariantEngine(vs)
    img = load_rgb(images[0][1])
    for v in vs:
        r = eng.run_image(img, v)
        assert 0 <= r["index"] < 1000
        assert 0 <= r["margin"] <= r["confidence"] <= 1.0 + 1e-6
        assert r["latency_ms"] > 0
