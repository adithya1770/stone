import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

TINY = "datasets/tiny-imagenet-200/val"


def sample_images(n=6):
    from v2.common.datasets import tiny_imagenet_val
    if not os.path.isdir(TINY):
        pytest.skip("tiny-imagenet not available")
    return tiny_imagenet_val(limit=n, seed=3)


@pytest.fixture(scope="session")
def images():
    return sample_images()
