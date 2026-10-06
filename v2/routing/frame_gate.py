"""Frame reuse for the live camera server.

web_demo's page sends a camera frame every 1.5 s. Consecutive frames of a
static scene are near-identical, so re-classifying them wastes compute and
energy. FrameGate compares a tiny 16x16 grayscale signature of the new
frame with the last classified frame; if the mean absolute difference is
below `threshold` (0-255 scale) and the cached result is younger than
`max_age_s`, the cached prediction is returned without inference.

This is reported separately from the model-selection results, so it never
inflates the comparison with the original controllers.
"""
import time

import numpy as np


class FrameGate:
    def __init__(self, threshold=6.0, max_age_s=5.0, size=16, enabled=True):
        self.threshold = threshold
        self.max_age_s = max_age_s
        self.size = size
        self.enabled = enabled
        self._sig = None
        self._result = None
        self._t = 0.0
        self.reused = 0
        self.total = 0

    def signature(self, img):
        return np.asarray(img.convert("L").resize((self.size, self.size)), dtype=np.float32)

    def check(self, img):
        """Return (cached_result_or_None, diff, signature)."""
        self.total += 1
        sig = self.signature(img)
        if not self.enabled or self._sig is None:
            return None, None, sig
        diff = float(np.abs(sig - self._sig).mean())
        if diff < self.threshold and (time.time() - self._t) <= self.max_age_s:
            self.reused += 1
            return self._result, diff, sig
        return None, diff, sig

    def store(self, sig, result):
        self._sig, self._result, self._t = sig, result, time.time()
