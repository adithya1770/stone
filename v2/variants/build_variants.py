"""Build correctly-preprocessed model variants (optional, needs internet once).

The original models/ files have two issues (see v2/README.md):
  - mobilenet_v2_fp32.tflite is dynamic-range quantized (int8 weights),
    not true FP32;
  - the original engine feeds [0,1] inputs, while Keras MobileNetV2 expects
    [-1,1]; the INT8 model was most likely CALIBRATED on [0,1] as well, so
    its accuracy cannot be fixed by changing preprocessing alone.

This script rebuilds the models from the official Keras ImageNet weights
with the normalisation BAKED INTO the graph, so every model takes the same
[0,1] input as the original engine (no special cases anywhere):

  v2/variants/mnv2_fp32.tflite    true float32 MobileNetV2      kind 'mnv2f'
  v2/variants/mnv2_int8.tflite    full-integer, correct calib.  kind 'mnv2q'
  v2/variants/mnv3s_fp32.tflite   MobileNetV3-Small float32     kind 'mnv3sf'
  v2/variants/mnv3s_int8.tflite   MobileNetV3-Small int8        kind 'mnv3sq'
  v2/variants/registry.json       -> picked up by common/engine.py

Run on a machine with internet (your Mac or the Pi):
    pip install tensorflow
    python -m v2.variants.build_variants                 # calibrates on imagenette train if present, else tiny-imagenet train
Then use variants such as mnv2q_160, mnv3sq_224 in the benchmarks
(they are added to the default ladder automatically once registry.json exists).

--random-weights builds the same graphs with random weights (no download):
only for testing this script.
"""
import argparse
import glob
import json
import os
import random

import numpy as np
from PIL import Image

from v2.common.paths import V2_ROOT, assert_inside_v2, enter_repo

OUT = os.path.join(V2_ROOT, "variants")


def calib_images(n, seed=0, calib_dir=None):
    pats = ["datasets/imagenette2-320/train/*/*.JPEG", "datasets/tiny-imagenet-200/train/*/images/*.JPEG"]
    if calib_dir:
        pats = [os.path.join(calib_dir, "**", "*.[jJ][pP][gG]"), os.path.join(calib_dir, "**", "*.[jJ][pP][eE][gG]")]
    for p in pats:
        files = sorted(glob.glob(p, recursive=True))
        if files:
            random.Random(seed).shuffle(files)
            print(f"calibration: {min(n, len(files))} images matching {p} (never used for evaluation)")
            return files[:n]
    raise SystemExit("no calibration images found (need datasets/imagenette2-320/train or tiny-imagenet-200/train)")


def build_keras(arch, random_weights):
    import tensorflow as tf
    weights = None if random_weights else "imagenet"
    inp = tf.keras.Input((224, 224, 3), name="image_0_1")       # same input range as the original engine
    if arch == "mnv2":
        x = tf.keras.layers.Rescaling(2.0, offset=-1.0)(inp)      # [0,1] -> [-1,1] (Keras MobileNetV2 convention)
        base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), weights=weights, classifier_activation="softmax")
    elif arch == "mnv3s":
        x = tf.keras.layers.Rescaling(255.0)(inp)                 # [0,1] -> [0,255]; V3 rescales internally
        base = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3), weights=weights,
                                                      include_preprocessing=True, classifier_activation="softmax")
    else:
        raise ValueError(arch)
    return tf.keras.Model(inp, base(x), name=f"{arch}_0_1")


def convert(model, int8, calib):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        def rep():
            for p in calib:
                img = Image.open(p).convert("RGB").resize((224, 224))
                yield [np.expand_dims((np.array(img) / 255.0).astype(np.float32), 0)]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["mnv2", "mnv3s"])
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--calib-dir", default=None, help="folder of calibration images (default: imagenette train, else tiny-imagenet train)")
    ap.add_argument("--random-weights", action="store_true", help="test mode: no download, random weights")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    enter_repo()
    out = assert_inside_v2(a.out)
    os.makedirs(out, exist_ok=True)
    calib = calib_images(a.n_calib, calib_dir=a.calib_dir)
    kinds = {"mnv2": ("mnv2f", "mnv2q"), "mnv3s": ("mnv3sf", "mnv3sq")}
    reg_path = os.path.join(out, "registry.json")
    reg = json.load(open(reg_path)) if os.path.exists(reg_path) else {}
    for arch in a.archs:
        model = build_keras(arch, a.random_weights)
        for int8, kind in zip((False, True), kinds[arch]):
            fn = f"{arch}_{'int8' if int8 else 'fp32'}{'_RANDOM' if a.random_weights else ''}.tflite"
            data = convert(model, int8, calib)
            with open(os.path.join(out, fn), "wb") as f:
                f.write(data)
            reg[kind + ("_random" if a.random_weights else "")] = {"path": fn, "float_range": "0_1"}
            print(f"  {kind:<7} -> {fn}  ({len(data) / 1e6:.1f} MB)")
    with open(reg_path, "w") as f:
        json.dump(reg, f, indent=2)
    print(f"registry -> {reg_path}")


if __name__ == "__main__":
    main()
