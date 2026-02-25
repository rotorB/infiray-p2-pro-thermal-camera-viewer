"""
Thermal matrix de-noise for microbolometer cameras (e.g. Infiray P2 Pro).

Two-stage pipeline:
  1. Temporal EMA — averages consecutive frames to cancel random per-pixel
     fluctuations.  Zero spatial blur.
  2. Spatial Non-Local Means (NLM) — finds similar patches across the image
     and averages only those, preserving edges far better than Gaussian or
     bilateral filters.  Kicks in only above strength 1.5.

Applied on float64 temperature array (Celsius) after conversion, before
normalization to 8-bit.  Single "strength" knob (1..10) controls both stages.
"""

import cv2
import numpy as np


class ThermalDenoiser:
    """
    Adjustable two-stage denoiser: temporal EMA + spatial NLM.

    strength (1..10):
      1   — almost pass-through (very light temporal only)
      3   — default: mild temporal + light NLM
      5   — moderate
      10  — heavy smoothing (static scenes)
    """

    STRENGTH_MIN = 1.0
    STRENGTH_MAX = 10.0
    STRENGTH_STEP = 0.5
    STRENGTH_DEFAULT = 3.0

    NLM_TEMPLATE = 7
    NLM_SEARCH = 21

    def __init__(self, shape):
        self.shape = shape
        self.strength = self.STRENGTH_DEFAULT
        self._accum = None

    def reset(self):
        """Discard temporal history."""
        self._accum = None

    @property
    def _temporal_alpha(self):
        """Weight of new frame in EMA.  Lower = heavier averaging."""
        return max(0.25, 1.0 - self.strength * 0.07)

    @property
    def _nlm_h(self):
        """NLM filter strength in uint8 space.  0 = spatial stage off."""
        return max(0.0, (self.strength - 1.5) * 1.5)

    def adjust(self, delta):
        """Change strength by delta, clamping to valid range."""
        self.strength = round(
            max(self.STRENGTH_MIN,
                min(self.STRENGTH_MAX, self.strength + delta)),
            1,
        )

    def apply(self, temp_celsius: np.ndarray) -> np.ndarray:
        """
        Denoise a single temperature frame (float64, H×W).
        Returns denoised float64 array, same shape.
        """
        frame = np.asarray(temp_celsius, dtype=np.float64)
        if frame.shape != self.shape:
            return frame

        # --- Stage 1: temporal EMA ---
        alpha = self._temporal_alpha
        if self._accum is None:
            self._accum = frame.copy()
        else:
            self._accum = alpha * frame + (1.0 - alpha) * self._accum

        result = self._accum

        # --- Stage 2: spatial NLM (only when strength > ~1.5) ---
        h = self._nlm_h
        if h > 0.1:
            t_min = np.min(result)
            t_range = max(np.ptp(result), 0.01)
            norm = ((result - t_min) / t_range * 255.0)
            norm = np.clip(norm, 0, 255).astype(np.uint8)

            denoised = cv2.fastNlMeansDenoising(
                norm, None, h,
                self.NLM_TEMPLATE, self.NLM_SEARCH,
            )

            result = denoised.astype(np.float64) / 255.0 * t_range + t_min

        return result
