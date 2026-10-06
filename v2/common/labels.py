"""Ground-truth label handling for STONE v2.

Correctness is decided by comparing class *indices* (0..999 in standard
ImageNet-1k order), not label strings. labels.txt contains duplicate
names (e.g. two different 'crane' classes), so string comparison can
mark a wrong prediction as correct. For the 10 Imagenette classes the
two methods agree, which tests/test_labels.py checks.

data/imagenet_synsets.txt is the standard ordered list of the 1000
ImageNet-1k WordNet IDs (taken from the Apache-2.0 'timm' package).
"""
import os

from .paths import DATA_ROOT

_SYNSETS = None


def synsets():
    global _SYNSETS
    if _SYNSETS is None:
        with open(os.path.join(DATA_ROOT, "imagenet_synsets.txt")) as f:
            _SYNSETS = [l.strip() for l in f if l.strip()]
        assert len(_SYNSETS) == 1000, len(_SYNSETS)
    return _SYNSETS


_WNID_INDEX = None


def wnid_to_index(wnid):
    """WordNet id -> ImageNet-1k class index (0..999), or None."""
    global _WNID_INDEX
    if _WNID_INDEX is None:
        _WNID_INDEX = {w: i for i, w in enumerate(synsets())}
    return _WNID_INDEX.get(wnid)


def load_labels(path="models/labels.txt"):
    with open(path) as f:
        return [line.strip() for line in f]


def label_offset(labels, n_outputs):
    """labels.txt has 1001 lines ('background' first) while the models
    output 1000 classes -> offset 1, same as the original engine's +1."""
    return len(labels) - n_outputs
