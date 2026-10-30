"""
Thermal matrix de-noise for microbolometer cameras (e.g. Infiray P2 Pro).

Problem: neighboring pixels on the same surface should share temperature, but
sensor NETD noise makes them flicker. Real thermal edges (object boundaries)
must stay sharp.

Algorithm:
  1. Motion-compensated Kalman — warp the previous estimate onto this frame
     (NCC translation), then a per-pixel Kalman update. Static regions
     collapse variance (~1/sqrt(N)); pixels that don't match the warp keep
     the new measurement (no ghost trail).
  2. Estimate noise σ from MAD of a high-pass residual.
  3. Spatial Edge-Gated Local Mean: average only neighbors whose temperature
     differs by less than ~k·σ, so real edges are not smeared.

Applied on float temperature (°C) after conversion, before 8-bit mapping.
Single strength knob (1..10) controls both stages.
"""

from __future__ import annotations

import cv2
import numpy as np

from fpn_correction import _downsample, _ncc_shift


def estimate_noise_sigma(frame: np.ndarray) -> float:
    """
    Robust noise estimate: MAD of high-pass residual.
    Laplacian-like residual via frame - mild Gaussian blur.
    """
    blur = cv2.GaussianBlur(frame, (0, 0), 0.8)
    residual = frame - blur
    mad = float(np.median(np.abs(residual)))
    # Gaussian noise: σ ≈ 1.4826 · MAD
    return max(1.4826 * mad, 1e-4)


def edge_gated_local_mean(
    frame: np.ndarray,
    radius: int,
    n_sigma: float,
    noise_sigma: float | None = None,
) -> np.ndarray:
    """
    Local mean over temperature-homogeneous neighbors only.

    For each pixel p, collect q in a (2r+1)×(2r+1) window where
        |T(q) - T(p)| <= n_sigma * σ_noise
    and replace T(p) with the distance-weighted mean of those q.
    Pixels across a real thermal edge fall outside the gate and are ignored.

    Args:
        frame: float HxW temperature map (°C)
        radius: spatial window radius in pixels (1..4 typical)
        n_sigma: homogeneity gate width in units of noise σ
        noise_sigma: optional precomputed noise σ; estimated if None
    """
    img = np.asarray(frame, dtype=np.float32)
    h, w = img.shape
    if radius < 1:
        return img

    if noise_sigma is None:
        noise_sigma = estimate_noise_sigma(img)
    thresh = float(n_sigma) * float(noise_sigma)

    padded = np.pad(img, radius, mode="edge")

    # Stack all shifted neighborhoods: (N, H, W)
    shifts = []
    spatial_w = []
    inv_2s2 = 1.0 / (2.0 * max(radius * 0.6, 0.5) ** 2)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0 = radius + dy
            x0 = radius + dx
            shifts.append(padded[y0 : y0 + h, x0 : x0 + w])
            # Mild spatial preference for closer neighbors
            spatial_w.append(np.exp(-(dx * dx + dy * dy) * inv_2s2))

    stack = np.stack(shifts, axis=0)  # (N, H, W)
    spat = np.asarray(spatial_w, dtype=np.float32)[:, None, None]

    # Homogeneity gate relative to center pixel
    center = img[None, ...]
    gate = (np.abs(stack - center) <= thresh).astype(np.float32)
    weights = gate * spat

    # The center pixel always passes its own gate with weight 1.0, so wsum >= 1
    # and the mean is always well defined; no fallback branch is needed.
    wsum = weights.sum(axis=0)
    out = (stack * weights).sum(axis=0) / wsum
    return out.astype(np.float32)


def _translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """dst[y, x] = src[y+dy, x+dx] with subpixel linear sampling."""
    h, w = img.shape[:2]
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


class ThermalDenoiser:
    """
    Adjustable two-stage denoiser: MC-Kalman + Edge-Gated Local Mean.

    strength (1..10):
      1   — almost pass-through (very light temporal only)
      3   — default: mild Kalman + light EGLM
      5   — moderate (good for the grainy P2 Pro look)
      10  — heavy smoothing (static / slow scenes)
    """

    STRENGTH_MIN = 1.0
    STRENGTH_MAX = 10.0
    STRENGTH_STEP = 0.5
    STRENGTH_DEFAULT = 3.0

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.strength = self.STRENGTH_DEFAULT
        self._x = None       # Kalman mean (HxW)
        self._p = None       # Kalman variance (HxW)
        self._prev = None    # last observation, for NCC

    def reset(self):
        """Discard temporal history."""
        self._x = None
        self._p = None
        self._prev = None

    @property
    def _q_scale(self) -> float:
        """Process-noise scale vs measurement R. Lower = heavier averaging."""
        return max(0.04, 1.55 - self.strength * 0.15)

    @property
    def _radius(self) -> int:
        """Spatial window radius grows with strength."""
        # 1→1, ~3→2, ~6→3, 10→4
        return int(np.clip(round(0.5 + self.strength * 0.35), 1, 4))

    @property
    def _n_sigma(self) -> float:
        """
        Homogeneity gate width. Larger = more neighbors averaged (smoother),
        but still stops at edges larger than ~n_sigma·noise.
        """
        # strength 1 → ~1.5σ, 5 → ~2.8σ, 10 → ~4.2σ
        return 1.2 + self.strength * 0.3

    @property
    def _spatial_iters(self) -> int:
        """Repeat EGLM for stronger smoothing without enlarging the gate."""
        if self.strength < 2.0:
            return 0  # temporal only
        if self.strength < 6.0:
            return 1
        if self.strength < 9.0:
            return 2
        return 3

    def adjust(self, delta: float) -> None:
        """Change strength by delta, clamping to valid range."""
        self.strength = round(
            max(self.STRENGTH_MIN, min(self.STRENGTH_MAX, self.strength + delta)),
            1,
        )

    def _kalman(self, z: np.ndarray) -> np.ndarray:
        """Warp previous state onto this frame, then per-pixel Kalman update."""
        sigma = estimate_noise_sigma(z)
        R = max(sigma * sigma, 1e-6)

        if self._x is None:
            self._x = z.copy()
            self._p = np.full(self.shape, R * 4.0, dtype=np.float32)
            self._prev = z.copy()
            return z

        small_prev = _downsample(self._prev)
        small_cur = _downsample(z)
        dy, dx, peak, on_border = _ncc_shift(small_prev, small_cur, radius=8)
        texture = float(small_cur.std())

        if peak >= 0.40 and not on_border:
            x_pred = _translate(self._x, dx, dy)
            p_pred = _translate(self._p, dx, dy)
        else:
            x_pred = self._x
            p_pred = self._p

        innov = z - x_pred
        # Extra process noise where the prediction missed (occlusion / new object).
        # On a low-texture wall, peak is unreliable — keep Q small and average.
        q_base = R * self._q_scale
        if texture < 0.4 and peak < 0.40:
            q_extra = 0.0
        else:
            q_extra = 0.25 * innov * innov
        q = q_base + q_extra
        p_pred = p_pred + q
        k = p_pred / (p_pred + R)
        self._x = x_pred + k * innov
        self._p = (1.0 - k) * p_pred
        self._prev = z.copy()
        return self._x

    def apply(self, temp_celsius: np.ndarray) -> np.ndarray:
        """
        Denoise a single temperature frame (float, H×W).

        Returns float32: the rest of the pipeline is float32, and float64 here
        only doubled the memory traffic without buying any precision (sensor
        resolution is ~1/64 K).
        """
        frame = np.asarray(temp_celsius, dtype=np.float32)
        if frame.shape != self.shape:
            return frame

        # --- Stage 1: motion-compensated Kalman ---
        result = self._kalman(frame)

        # --- Stage 2: Edge-Gated Local Mean ---
        iters = self._spatial_iters
        if iters > 0:
            sigma = estimate_noise_sigma(result)
            radius = self._radius
            n_sigma = self._n_sigma
            for _ in range(iters):
                result = edge_gated_local_mean(
                    result, radius=radius, n_sigma=n_sigma, noise_sigma=sigma
                )
                # Re-estimate lightly so later passes don't over-smooth
                sigma = max(sigma * 0.85, 1e-4)

        return result
