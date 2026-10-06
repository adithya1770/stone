"""Multi-variant TFLite inference engine for STONE v2.

A *variant* is "<kind>_<resolution>", e.g. "int8_128" or "fp32_224".
  - kind 'fp32' -> models/mobilenet_v2_fp32.tflite  (NB: this file is
    dynamic-range quantized: int8 weights, float activations)
  - kind 'int8' -> models/mobilenet_v2_int8.tflite  (full integer)
  - extra kinds can be registered in v2/variants/registry.json
    (written by v2/variants/build_variants.py).

Resolution variants re-use the SAME weights: MobileNetV2 is fully
convolutional with a global pool, so TFLite's resize_tensor_input lets one
model run at 224/192/160/128/96 px. That gives a real latency ladder
without downloading anything.

Preprocessing at 224 px is bit-identical to runtime/inference_engine.py
(checked by tests/test_engine.py), so v2's FP32/INT8 numbers are directly
comparable with the original results.
"""
import json
import os
import time

import numpy as np
from PIL import Image

import ai_edge_litert.interpreter as tflite

from .labels import label_offset, load_labels
from .paths import V2_ROOT

# kind -> (model path, float input range or None=engine default)
#   fp32  : original model + ORIGINAL preprocessing (img/255)  -> parity with v1
#   fp32n : same model file + Keras MobileNetV2 normalisation (img/127.5-1)
#           (see bench/check_preprocessing.py; reported as a separate ablation)
BASE_KINDS = {
    "fp32": ("models/mobilenet_v2_fp32.tflite", "0_1"),
    "fp32n": ("models/mobilenet_v2_fp32.tflite", "-1_1"),
    "int8": ("models/mobilenet_v2_int8.tflite", None),
}

# Ladder used by default in benchmarks (cheapest -> most expensive).
DEFAULT_VARIANTS = [
    "int8_96", "int8_128", "int8_160", "int8_192", "int8_224",
    "fp32_128", "fp32_160", "fp32_224",
    "fp32n_128", "fp32n_160", "fp32n_224",
]
ORIGINAL_PREPROC_VARIANTS = [v for v in DEFAULT_VARIANTS if not v.startswith("fp32n")]


def default_variants(resolutions=(128, 160, 224)):
    """DEFAULT_VARIANTS + every kind registered by variants/build_variants.py
    (skipping test-only '*_random' kinds)."""
    kinds = load_registry()
    extra = [k for k in kinds if k not in BASE_KINDS and not k.endswith("_random") and k not in NATIVE_RES]
    return DEFAULT_VARIANTS + [f"{k}_{r}" for k in extra for r in resolutions]


def native_variants(kinds=None):
    """Downloaded models at their native input size, e.g. ['efl0q_224', ..., 'efl4q_300']."""
    load_registry()
    return [f"{k}_{r}" for k, r in NATIVE_RES.items() if kinds is None or k in kinds]


# float input conventions: x = (pixel - mean) / std
FLOAT_RANGES = {
    "0_1": (0.0, 255.0),        # original engine (img/255); rebuilt models (normalisation baked in)
    "-1_1": (127.5, 127.5),     # Keras MobileNetV2 (img/127.5 - 1)
    "effnet": (127.0, 128.0),   # EfficientNet-Lite (official TFLite models: (img - 127) / 128)
}

# registries: v2/variants (models we build) and v2/models_ext (downloaded official models)
REGISTRIES = ("variants", "models_ext")
NATIVE_RES = {}   # kind -> native input size, for downloaded models (filled by load_registry)


def load_registry():
    kinds = dict(BASE_KINDS)
    for sub in REGISTRIES:
        reg = os.path.join(V2_ROOT, sub, "registry.json")
        if not os.path.exists(reg):
            continue
        with open(reg) as f:
            for k, spec in json.load(f).items():
                p, rng = (spec, None) if isinstance(spec, str) else (spec["path"], spec.get("float_range"))
                path = p if os.path.isabs(p) else os.path.join(V2_ROOT, sub, p)
                if os.path.exists(path):
                    kinds[k] = (path, rng)
                    if isinstance(spec, dict) and spec.get("native"):
                        NATIVE_RES[k] = int(spec["native"])
    return kinds


def parse_variant(name):
    kind, res = name.rsplit("_", 1)
    return kind, int(res)


def load_rgb(path):
    """Decode once; cascades re-use the decoded image for the 2nd stage."""
    return Image.open(path).convert("RGB")


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def scores_from_output(output):
    """Return (probs, confidence, margin, entropy, top1, top2).

    The models end in softmax already; if an output does not look like a
    probability vector (e.g. a logits model from the builder) apply one.
    confidence = max prob (exactly what the original engine logs).
    margin = p1 - p2: how clearly the top guess beats the runner-up.
    """
    out = output.astype(np.float64)
    s = out.sum()
    probs = out if (out.min() >= -1e-6 and 0.5 < s < 1.5) else _softmax(out)
    top2 = np.argpartition(probs, -2)[-2:]
    top2 = top2[np.argsort(probs[top2])[::-1]]
    p1, p2 = float(probs[top2[0]]), float(probs[top2[1]])
    pn = probs[probs > 1e-12] / max(probs.sum(), 1e-12)
    entropy = float(-(pn * np.log(pn)).sum())
    return probs, p1, p1 - p2, entropy, int(top2[0]), int(top2[1])


class MultiVariantEngine:
    def __init__(self, variants=None, labels_path="models/labels.txt",
                 num_threads=None, float_range="0_1"):
        """num_threads=None keeps the TFLite default, matching the original
        engine. float_range: default float input range for kinds that do not
        fix their own ('0_1' = original img/255, '-1_1' = img/127.5-1)."""
        self.kinds = load_registry()
        self.labels = load_labels(labels_path)
        self.num_threads = num_threads
        self.float_range = float_range
        self._interp = {}
        for v in (variants or []):
            self.get(v)

    # ---- model management -------------------------------------------
    def get(self, variant):
        if variant not in self._interp:
            kind, res = parse_variant(variant)
            if kind not in self.kinds:
                raise KeyError(f"unknown model kind '{kind}' in variant '{variant}'")
            path, frange = self.kinds[kind]
            kw = {"model_path": path}
            if self.num_threads is not None:
                kw["num_threads"] = self.num_threads
            it = tflite.Interpreter(**kw)
            inp = it.get_input_details()[0]
            if tuple(inp["shape"][1:3]) != (res, res):
                it.resize_tensor_input(inp["index"], [1, res, res, 3])
            it.allocate_tensors()
            inp = it.get_input_details()[0]
            out = it.get_output_details()[0]
            self._interp[variant] = {
                "it": it, "inp": inp, "out": out, "res": res, "kind": kind,
                "offset": label_offset(self.labels, int(np.prod(out["shape"][1:]))),
                "float_range": frange or self.float_range,
            }
            self._warmup(variant)
        return self._interp[variant]

    def loaded(self):
        return list(self._interp)

    def _warmup(self, variant):
        m = self._interp[variant]
        m["it"].set_tensor(m["inp"]["index"], np.zeros(m["inp"]["shape"], dtype=m["inp"]["dtype"]))
        m["it"].invoke()

    # ---- preprocessing ------------------------------------------------
    def preprocess(self, img, variant):
        m = self.get(variant)
        res = m["res"]
        arr = np.array(img.resize((res, res)))  # same call as original engine
        dt = m["inp"]["dtype"]
        fr = m["float_range"] if m["float_range"] in FLOAT_RANGES else "0_1"
        mean, std = FLOAT_RANGES[fr]
        if dt == np.float32:
            if fr == "-1_1":
                x = (arr / 127.5 - 1.0).astype(np.float32)
            elif fr == "0_1":
                x = (arr / 255.0).astype(np.float32)
            else:
                x = ((arr - mean) / std).astype(np.float32)
        else:
            q = m["inp"]["quantization_parameters"]
            scale = float(q["scales"][0]) if len(q["scales"]) else 1.0
            zp = int(q["zero_points"][0]) if len(q["zero_points"]) else 0
            if fr == "0_1" and dt == np.int8 and abs(scale - 1 / 255) < 1e-6 and zp == -128:
                # exact original formula: (img - 128) as int8
                x = (arr.astype(np.int32) - 128).astype(np.int8)
            else:
                # quantize the float input the model was trained/calibrated on
                info = np.iinfo(dt)
                x = np.clip(np.round(((arr - mean) / std) / scale) + zp, info.min, info.max).astype(dt)
        return np.expand_dims(x, 0)

    # ---- inference ----------------------------------------------------
    def run_image(self, img, variant):
        """Run one decoded PIL image through one variant."""
        m = self.get(variant)
        t0 = time.perf_counter()
        x = self.preprocess(img, variant)
        t1 = time.perf_counter()
        m["it"].set_tensor(m["inp"]["index"], x)
        t2 = time.perf_counter()
        m["it"].invoke()
        t3 = time.perf_counter()
        raw = m["it"].get_tensor(m["out"]["index"]).flatten()
        if m["out"]["dtype"] != np.float32:
            q = m["out"]["quantization_parameters"]
            raw = float(q["scales"][0]) * (raw.astype(np.float32) - int(q["zero_points"][0]))
        _, conf, margin, ent, top1, top2 = scores_from_output(raw)
        li = top1 + m["offset"]
        return {
            "variant": variant,
            "index": top1,                      # ImageNet-1k class index
            "label": self.labels[li] if 0 <= li < len(self.labels) else "unknown",
            "confidence": round(conf, 4),
            "margin": round(margin, 4),
            "entropy": round(ent, 4),
            "latency_ms": round((t3 - t2) * 1000, 3),        # invoke only (original definition)
            "preprocess_ms": round((t1 - t0) * 1000, 3),
        }

    def run(self, path, variant):
        t0 = time.perf_counter()
        img = load_rgb(path)
        r = self.run_image(img, variant)
        r["decode_ms"] = round((time.perf_counter() - t0) * 1000 - r["latency_ms"] - r["preprocess_ms"], 3)
        return r
