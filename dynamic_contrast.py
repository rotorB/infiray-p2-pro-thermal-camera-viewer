"""Temperature-coded hue with bounded, spatial-only lightness enhancement.

This is display processing: input Celsius samples are never modified. Oklab
gamut mapping is cached in a small LUT, avoiding per-frame RGB conversions.
"""

from functools import lru_cache
import time

import cv2
import numpy as np


def _linear_rgb(lightness, a, b):
    l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return np.stack((4.0767416621*l - 3.3077115913*m + 0.2309699292*s,
                     -1.2684380046*l + 2.6097574011*m - 0.3413193965*s,
                     -0.0041960863*l - 0.7034186147*m + 1.7076147010*s), axis=-1)


@lru_cache(maxsize=4)
def _palette_lut(flag):
    rgb = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], flag)[:, 0, ::-1]
    rgb = rgb.astype(np.float32) / 255.0
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                       [0.2119034982, 0.6806995451, 0.1073969566],
                       [0.0883024619, 0.2817188376, 0.6299787005]], dtype=np.float32)
    l, m, s = np.cbrt(rgb @ matrix.T).T
    lightness = 0.2104542553*l + 0.793617785*m - 0.0040720468*s
    a = (1.9779984951*l - 2.428592205*m + 0.4505937099*s)[:, None]
    b = (0.0259040371*l + 0.7827717662*m - 0.808675766*s)[:, None]
    # Leave room for details both below and above every palette color.
    lightness = (0.68 + 0.20 * (lightness[:, None] - 0.68)
                 + np.linspace(-0.20, 0.20, 257, dtype=np.float32)[None, :])
    # Reduce chroma along a constant-hue ray to fit sRGB; do not clip channels
    # independently, which would shift the temperature-coded hue.
    lo = np.zeros_like(lightness)
    hi = np.ones_like(lightness)
    for _ in range(11):
        chroma = (lo + hi) * 0.5
        candidate = _linear_rgb(lightness, a * chroma, b * chroma)
        valid = np.all((candidate >= 0.0) & (candidate <= 1.0), axis=-1)
        lo = np.where(valid, chroma, lo)
        hi = np.where(valid, hi, chroma)
    rgb = np.clip(_linear_rgb(lightness, a * lo, b * lo), 0.0, 1.0)
    rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1.0 / 2.4) - 0.055)
    return np.ascontiguousarray(np.rint(rgb[..., ::-1] * 255.0).astype(np.uint8))


def _edge_aware_base(src, radius):
    """Self-guided filter in Celsius; large thermal edges stay in the base."""
    size = (2 * radius + 1,) * 2
    mean = cv2.blur(src, size)
    variance = np.maximum(cv2.blur(src * src, size) - mean * mean, 0.0)
    a = variance / (variance + 0.49)  # 0.7 °C edge scale
    return cv2.blur(a, size) * src + cv2.blur((1.0 - a) * mean, size)


class DynamicContrast:
    def __init__(self):
        self._range = None
        self._range_time = 0.0
        self.render_ms = 0.0

    def reset_range(self):
        self._range = None

    def stabilize_range(self, low, high):
        """Smooth only two display endpoints, never past image content."""
        now = time.monotonic()
        dt = now - self._range_time
        if self._range is None or dt > 0.5:
            self._range = (low, high)
        else:
            alpha = 1.0 - np.exp(-max(dt, 0.001) / 0.25)
            old_low, old_high = self._range
            if abs(low - old_low) + abs(high - old_high) > 3.0 * max(old_high - old_low, 1.0):
                alpha = 1.0
            self._range = (old_low + alpha * (low - old_low),
                           old_high + alpha * (high - old_high))
        self._range_time = now
        return self._range

    def prepare(self, palette_flag):
        _palette_lut(palette_flag)

    def render(self, temp, low, high, palette_flag, *, gain=2.0, invert=False):
        start = time.perf_counter()
        src = np.asarray(temp, dtype=np.float32)
        # Native-size edge-aware bases are cheap; HR detail is retained by
        # subtracting their interpolated base from the original float field.
        h, w = src.shape
        small_size = (min(w, 256), min(h, 192))
        small = cv2.resize(src, small_size, interpolation=cv2.INTER_AREA)
        center = float(small.mean())
        small = small - center  # avoid cancellation for high absolute °C
        base = 0.65 * _edge_aware_base(small, 4) + 0.35 * _edge_aware_base(small, 12)
        base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
        detail = (src - center) - base
        # Soft noise gate, followed by a bounded gain. No temporal image blend.
        square = detail * detail
        detail *= square / (square + 0.06 ** 2)
        strength = np.tanh(detail * (float(np.clip(gain, 0.0, 4.0)) / 0.5))
        detail_index = np.rint((strength + 1.0) * 128.0).astype(np.intp)
        gray = np.clip((src - low) * (255.0 / max(high - low, 0.01)), 0, 255).astype(np.uint8)
        if invert:
            gray = 255 - gray
        result = _palette_lut(palette_flag)[gray, detail_index]
        self.render_ms = (time.perf_counter() - start) * 1000.0
        return result
