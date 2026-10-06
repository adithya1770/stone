"""Image sources for STONE v2 benchmarks.

Every source yields (image_id, path, true_index) where true_index is the
ImageNet-1k class index (or None if unknown).
"""
import hashlib
import os
import random

from .labels import wnid_to_index

IMAGENETTE_VAL = "datasets/imagenette2-320/val"
TINY_VAL = "datasets/tiny-imagenet-200/val"
IMAGEWOOF_VAL = "datasets/imagewoof2-320/val"   # 10 dog breeds (harder than Imagenette), same layout


def his_pool(n=100, seed=42):
    """The exact 100-image sequence the original experiments used
    (runtime/image_pool.ImagePool, seed 42, fixed order)."""
    from runtime.image_pool import ImagePool  # original code, read-only
    pool = ImagePool(seed=seed)
    items = []
    for i in range(min(n, len(pool))):
        p = pool.get(i)
        items.append((p, p, wnid_to_index(os.path.basename(os.path.dirname(p)))))
    return items


def imagefolder(root, limit=None, seed=0, exclude=()):
    """<root>/<wnid>/<img> layout (Imagenette)."""
    exclude = set(exclude)
    items = []
    for wnid in sorted(os.listdir(root)):
        d = os.path.join(root, wnid)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                p = os.path.join(d, f)
                if p not in exclude:
                    items.append((p, p, wnid_to_index(wnid)))
    random.Random(seed).shuffle(items)
    return items[:limit] if limit else items


def tiny_imagenet_val(root=TINY_VAL, limit=None, seed=0):
    """Tiny-ImageNet val (64x64 source images). Only classes that exist in
    ImageNet-1k are kept. NOTE: sources are 64 px, so they under-state the
    accuracy cost of low-resolution variants - use for pipeline checks and
    routing-mechanism tests, not for headline accuracy."""
    items = []
    with open(os.path.join(root, "val_annotations.txt")) as f:
        for line in f:
            parts = line.split("\t")
            idx = wnid_to_index(parts[1])
            if idx is not None:
                p = os.path.join(root, "images", parts[0])
                items.append((p, p, idx))
    random.Random(seed).shuffle(items)
    return items[:limit] if limit else items


def get_source(name, limit=None, seed=0, exclude_his_pool=False, exclude=()):
    """exclude: image ids/paths that must not be returned."""
    exclude = set(exclude)
    if name == "his_pool":
        return his_pool()
    if name == "imagenette":
        if exclude_his_pool:
            exclude |= {p for p, _, _ in his_pool()}
        return imagefolder(IMAGENETTE_VAL, limit, seed, exclude)
    if name == "imagenette_fresh":
        # never seen by the original protocol (its 150-image pool) nor by v2
        # tuning (pass the response table via exclude)
        exclude |= set(__import__("runtime.image_pool", fromlist=["ImagePool"]).ImagePool(seed=42).pool)
        return imagefolder(IMAGENETTE_VAL, limit, seed, exclude)
    if name == "imagewoof":
        return imagefolder(IMAGEWOOF_VAL, limit, seed, exclude)
    if name in ("imagewoof_tune", "imagewoof_test"):
        # hash halves of Imagewoof val: thresholds are tuned on one half, live runs use the other
        half = name.split("_")[1]
        items = [it for it in imagefolder(IMAGEWOOF_VAL, None, seed, exclude) if split_of(it[0]) == half]
        return items[:limit] if limit else items
    if name == "tiny":
        items = tiny_imagenet_val(seed=seed)
        items = [it for it in items if it[0] not in exclude]
        return items[:limit] if limit else items
    if os.path.isdir(name):
        return imagefolder(name, limit, seed)
    raise ValueError(f"unknown dataset '{name}'")


def split_of(image_id, tune_frac=0.5):
    """Deterministic tune/test split by hash, so tuning never sees test images."""
    h = int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "tune" if h < tune_frac else "test"
