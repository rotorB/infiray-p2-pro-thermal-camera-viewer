"""
Multi-frame super-resolution from subpixel shifts, in temperature space.

Live path: shift-and-add of the last N float °C frames.
HQ path (snapshot): sigma-clipped fusion + IBP + mild Richardson–Lucy.

Shifts are estimated on a blurred copy so residual FPN does not pin the
peak at zero (phase correlation's whitening would otherwise lock onto the
fixed pattern). Huge hops are rejected and the burst is re-anchored.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

def _gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k.astype(np.float32)


def richardson_lucy(img: np.ndarray, psf_sigma: float = 0.7, n_iter: int = 6) -> np.ndarray:
    """Mild RL deconvolution. Shifts the field to stay strictly positive."""
    img = np.ascontiguousarray(img, dtype=np.float32)
    offset = float(img.min()) - 1.0
    x = img - offset
    observed = x.copy()
    psf = _gaussian_kernel(psf_sigma)
    psf_flip = psf[::-1, ::-1].copy()
    for _ in range(max(0, int(n_iter))):
        conv = cv2.filter2D(x, -1, psf, borderType=cv2.BORDER_REFLECT)
        conv = np.maximum(conv, 1e-6)
        relative = observed / conv
        x = x * cv2.filter2D(relative, -1, psf_flip, borderType=cv2.BORDER_REFLECT)
        x = np.maximum(x, 1e-6)
    return x + offset


class BurstSR:
    """
    Accumulates LR float frames, estimates subpixel shifts, fuses onto an
    HR grid (scale × LR size).
    """

    MAX_SHIFT_PX = 8.0
    REANCHOR_AFTER = 3
    MIN_RESPONSE = 0.08

    def __init__(self, lr_shape, scale=3, max_frames=16):
        self.lr_shape = tuple(lr_shape)
        self.scale = int(scale)
        self.hr_shape = (self.lr_shape[0] * self.scale, self.lr_shape[1] * self.scale)
        self.max_frames = max(2, min(int(max_frames), 25))
        self._buffer = deque(maxlen=self.max_frames)
        self._ref = None
        self._rejects = 0

    def reset(self):
        self._buffer.clear()
        self._ref = None
        self._rejects = 0

    def _estimate_shift(self, ref, curr):
        """Subpixel shift of curr relative to ref, or None if unusable."""
        if ref.shape != curr.shape:
            return None
        a = cv2.GaussianBlur(np.ascontiguousarray(ref, dtype=np.float32), (0, 0), 1.4)
        b = cv2.GaussianBlur(np.ascontiguousarray(curr, dtype=np.float32), (0, 0), 1.4)
        try:
            (dx, dy), resp = cv2.phaseCorrelate(a, b)
        except cv2.error:
            return None
        if not np.isfinite(dx) or not np.isfinite(dy) or not np.isfinite(resp):
            return None
        if resp < self.MIN_RESPONSE:
            return None
        if abs(dx) > self.MAX_SHIFT_PX or abs(dy) > self.MAX_SHIFT_PX:
            return None
        return float(dx), float(dy)

    def push(self, lr_frame: np.ndarray) -> None:
        """Add one LR frame (H, W) float °C."""
        lr = np.ascontiguousarray(lr_frame, dtype=np.float32)
        if lr.shape[:2] != self.lr_shape:
            return
        if self._ref is None:
            self._ref = lr.copy()
            self._rejects = 0
            self._buffer.append((lr.copy(), 0.0, 0.0))
            return

        shift = self._estimate_shift(self._ref, lr)
        if shift is None:
            self._rejects += 1
            if self._rejects >= self.REANCHOR_AFTER:
                self._buffer.clear()
                self._ref = lr.copy()
                self._rejects = 0
                self._buffer.append((lr.copy(), 0.0, 0.0))
            return

        self._rejects = 0
        dx, dy = shift
        self._buffer.append((lr.copy(), dx, dy))

    def _remap(self, frame, dx, dy):
        h_hr, w_hr = self.hr_shape
        xx = np.arange(w_hr, dtype=np.float32) / self.scale + dx
        yy = np.arange(h_hr, dtype=np.float32) / self.scale + dy
        map_x = np.broadcast_to(xx, (h_hr, w_hr)).copy()
        map_y = np.broadcast_to(yy[:, np.newaxis], (h_hr, w_hr)).copy()
        return cv2.remap(
            frame, map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    def _fuse_mean(self):
        acc = np.zeros(self.hr_shape, dtype=np.float32)
        n = 0
        for frame, dx, dy in self._buffer:
            acc += self._remap(frame, dx, dy)
            n += 1
        return acc / max(n, 1)

    def get_hr(self):
        """Live fusion: mean of aligned frames. Float °C HR, or None if <2 frames."""
        if len(self._buffer) < 2:
            return None
        return self._fuse_mean()

    def get_hq(self):
        """
        Snapshot-quality fusion: sigma-clipped mean, IBP against the reference
        LR frame, mild RL deconv. None if the buffer is too thin.
        """
        if len(self._buffer) < 4:
            return self.get_hr()
        stack = np.stack(
            [self._remap(frame, dx, dy) for frame, dx, dy in self._buffer],
            axis=0,
        )
        med = np.median(stack, axis=0)
        mad = np.median(np.abs(stack - med), axis=0) + 1e-3
        cap = 2.5 * 1.4826 * mad
        fused = np.mean(np.clip(stack, med - cap, med + cap), axis=0)
        ref = self._ref if self._ref is not None else self._buffer[-1][0]
        # IBP so the fused HR still downscales to the (sharpest) reference.
        h, w = ref.shape[:2]
        hr_wh = (w * self.scale, h * self.scale)
        lo = float(ref.min())
        hi = float(ref.max())
        span = max(hi - lo, 1e-3)
        hr = fused
        for _ in range(6):
            recon = cv2.resize(hr, (w, h), interpolation=cv2.INTER_AREA)
            err = ref - recon
            err_hr = cv2.resize(err, hr_wh, interpolation=cv2.INTER_CUBIC)
            hr = hr + 0.45 * err_hr
            np.clip(hr, lo - 0.25 * span, hi + 0.25 * span, out=hr)
        return richardson_lucy(hr, psf_sigma=0.65, n_iter=5)

    @property
    def frame_count(self):
        return len(self._buffer)
