"""Path helpers for STONE v2.

v2 lives inside the original repo (stone/v2/) and imports the original
runtime/ package READ-ONLY. Nothing in v2 writes outside the v2/ folder.
"""
import os
import sys

V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(V2_ROOT)
RESULTS_ROOT = os.path.join(V2_ROOT, "results_v2")
CONFIG_ROOT = os.path.join(V2_ROOT, "configs")
DATA_ROOT = os.path.join(V2_ROOT, "data")


def enter_repo():
    """Put the repo root on sys.path and chdir into it.

    The original code uses repo-relative paths (models/..., datasets/...),
    so v2 scripts run from the repo root to stay compatible with it.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    os.chdir(REPO_ROOT)


def results_path(*parts):
    """Absolute path under v2/results_v2/, creating parent dirs."""
    p = os.path.join(RESULTS_ROOT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def assert_inside_v2(path):
    """Guard used before any write: refuse to write outside v2/."""
    ap = os.path.abspath(path)
    if not (ap == V2_ROOT or ap.startswith(V2_ROOT + os.sep)):
        raise PermissionError(f"v2 refuses to write outside v2/: {ap}")
    return ap
