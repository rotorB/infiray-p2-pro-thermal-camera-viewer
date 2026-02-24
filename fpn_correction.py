"""
Real-time Fixed-Pattern Noise (FPN) correction for thermal camera.

Estimates per-pixel offset (deviation from global mean) from a running average
over many frames, then subtracts this offset so the fixed pattern disappears.
The correction is applied in raw sensor units (before temperature conversion).
"""

import numpy as np


class FPNCorrector:
    """
    Maintains a per-pixel running mean and subtracts the offset map
    (pixel_mean - global_mean) from each frame to remove fixed-pattern noise.
    """

    def __init__(self, shape, alpha=0.995):
        """
        Args:
            shape: (height, width) of the thermal array, e.g. (192, 256).
            alpha: smoothing factor for exponential moving average.
                   Higher = slower adaptation (e.g. 0.995 = ~200 frames to converge).
        """
        self.shape = shape
        self.alpha = float(alpha)
        self._running_sum = None
        self._running_count = 0.0  # for stability in early frames

    def reset(self):
        """Reset calibration state (e.g. after scene change or user request)."""
        self._running_sum = None
        self._running_count = 0.0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Update internal running mean from frame, then return corrected frame.

        Args:
            frame: 2D array (e.g. uint16 raw thermal), shape must match self.shape.

        Returns:
            Corrected frame, same shape and dtype as input.
            If calibration is not yet ready, returns frame unchanged.
        """
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape != self.shape:
            return frame.astype(np.uint16) if frame.dtype != np.uint16 else np.asarray(frame, dtype=np.uint16)

        if self._running_sum is None:
            self._running_sum = frame.copy()
            self._running_count = 1.0
            return np.clip(frame, 0, 65535).astype(np.uint16)

        # Exponential moving average: running_mean = alpha * running_mean + (1-alpha) * frame
        self._running_sum = self.alpha * self._running_sum + (1.0 - self.alpha) * frame
        self._running_count = self.alpha * self._running_count + (1.0 - self.alpha)

        running_mean = self._running_sum / self._running_count
        global_mean = np.mean(running_mean)
        offset_map = running_mean - global_mean

        corrected = frame - offset_map
        corrected = np.clip(corrected, 0, 65535)
        return corrected.astype(np.uint16)
