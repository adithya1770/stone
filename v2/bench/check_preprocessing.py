"""Sanity check: is the original float preprocessing (img/255, range [0,1])
what models/mobilenet_v2_fp32.tflite expects, or the Keras MobileNetV2
convention (img/127.5 - 1, range [-1,1])?

The INT8 model's input quantization (scale 1/255, zero point -128) encodes
a [0,1] range, but that only reflects how it was calibrated, so the float
model is tested empirically on labelled images.

    python -m v2.bench.check_preprocessing --dataset imagenette --limit 500
"""
import argparse
import json
import os

from v2.common.paths import enter_repo, results_path
from v2.common.engine import MultiVariantEngine, load_rgb
from v2.common.datasets import get_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="auto")
    ap.add_argument("--limit", type=int, default=500)
    a = ap.parse_args()
    enter_repo()
    ds = a.dataset
    if ds == "auto":
        ds = "imagenette" if os.path.isdir("datasets/imagenette2-320/val") else "tiny"
    items = [it for it in get_source(ds, limit=a.limit, seed=7) if it[2] is not None]
    eng = MultiVariantEngine(["fp32_224", "fp32n_224"])
    ok01 = ok11 = 0
    for _, p, t in items:
        img = load_rgb(p)
        ok01 += eng.run_image(img, "fp32_224")["index"] == t
        ok11 += eng.run_image(img, "fp32n_224")["index"] == t
    n = len(items)
    res = {"dataset": ds, "n": n, "acc_0_1_original": round(100 * ok01 / n, 2),
           "acc_minus1_1": round(100 * ok11 / n, 2)}
    res["verdict"] = ("original [0,1] preprocessing is correct" if ok01 >= ok11
                      else "model expects [-1,1]: original float preprocessing under-reports FP32 accuracy")
    print(json.dumps(res, indent=2))
    with open(results_path("checks", f"preprocessing_{ds}.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
