"""
Thermal matrix de-noise for microbolometer cameras (e.g. Infiray P2 Pro).

Combines temporal averaging (exponential moving average across frames)
with edge-preserving spatial filtering (bilateral filter) to suppress
random noise and row/column artifacts while retaining thermal detail.

Applied on float64 temperature array (Celsius) after conversion, before
normalization to 8-bit.
"""

import cv2
import numpy as np


class ThermalDenoiser:
    """
    Two-stage denoiser: temporal EMA + spatial bilateral filter.

    Temporal stage smooths per-pixel random fluctuations across frames.
    Spatial stage removes residual salt-and-pepper and matrix grid noise
    while preserving object edges.
    """

    def __init__(self, shape, temporal_alpha=0.75, spatial_d=3,
                 spatial_sigma_color=0.8, spatial_sigma_space=0.8):
        """
        Args:
            shape: (height, width) of the temperature array.
            temporal_alpha: weight of new frame in EMA (0..1).
                0.75 = light smoothing (~2 frames), 0.3 = heavy (~5-6 frames).
            spatial_d: bilateral filter diameter (pixels). 3 = minimal footprint.
            spatial_sigma_color: bilateral color (temperature) sigma in Celsius.
                Lower = only smooth very similar temps, preserving all detail.
            spatial_sigma_space: bilateral spatial sigma in pixels.
        """
        self.shape = shape
        self.temporal_alpha = temporal_alpha
        self.spatial_d = spatial_d
        self.spatial_sigma_color = spatial_sigma_color
        self.spatial_sigma_space = spatial_sigma_space
        self._accum = None

    def reset(self):
        """Discard temporal history (e.g. after scene change)."""
        self._accum = None

    def apply(self, temp_celsius: np.ndarray) -> np.ndarray:
        """
        Denoise a temperature frame.

        Args:
            temp_celsius: float64 array (H, W) of temperatures in Celsius.

        Returns:
            Denoised float64 array, same shape.
        """
        frame = np.asarray(temp_celsius, dtype=np.float64)
        if frame.shape != self.shape:
            return frame

        # --- Temporal: exponential moving average ---
        if self._accum is None:
            self._accum = frame.copy()
        else:
            a = self.temporal_alpha
            self._accum = a * frame + (1.0 - a) * self._accum

        smoothed = self._accum

        # --- Spatial: bilateral filter (edge-preserving) ---
        # cv2.bilateralFilter needs 8-bit or 32-bit float input
        f32 = smoothed.astype(np.float32)
        filtered = cv2.bilateralFilter(
            f32, self.spatial_d,
            self.spatial_sigma_color,
            self.spatial_sigma_space,
        )

        return filtered.astype(np.float64)
