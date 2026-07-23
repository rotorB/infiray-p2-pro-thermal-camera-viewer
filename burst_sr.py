"""
Multi-frame super-resolution from subpixel shifts (Samsung-style burst SR).

Uses micro-movements between frames (camera shake or scene motion) to capture
the same scene at slightly different subpixel positions, then fuses frames
onto a higher-resolution grid. No learning; resolution comes from temporal
integration. Works with any image source including thermal.

Pipeline:
  1. Buffer last N LR frames (e.g. 9).
  2. Estimate subpixel shift of each frame vs reference (phase correlation).
  3. Fuse: for each HR pixel, sample each frame at corresponding LR position
     (accounting for shift), average. Result has more samples than sensor.
"""

import cv2
import numpy as np
from collections import deque


class BurstSR:
    """
    Accumulates LR frames, estimates subpixel shifts via phase correlation,
    fuses onto HR grid (scale x LR size).
    """

    def __init__(self, lr_shape, scale=3, max_frames=9):
        """
        Args:
            lr_shape: (H, W) of low-res input, e.g. (192, 256).
            scale: Output scale factor (e.g. 3 -> 576x768).
            max_frames: Max frames in buffer (4–16 typical).
        """
        self.lr_shape = lr_shape
        self.scale = scale
        self.hr_shape = (lr_shape[0] * scale, lr_shape[1] * scale)
        self.max_frames = max(2, min(max_frames, 25))
        self._buffer = deque(maxlen=self.max_frames)  # list of (frame_float32, dx, dy)
        self._ref = None  # reference frame for alignment

    def reset(self):
        self._buffer.clear()
        self._ref = None

    def _estimate_shift(self, ref, curr):
        """Subpixel shift of curr relative to ref. Returns (dx, dy) in pixels."""
        if ref.shape != curr.shape or ref.dtype != curr.dtype:
            return 0.0, 0.0
        try:
            # Phase correlation: shift of curr w.r.t. ref
            (dx, dy), _ = cv2.phaseCorrelate(
                np.asarray(ref, dtype=np.float32),
                np.asarray(curr, dtype=np.float32),
            )
            # Reject huge shifts (scene change or failure)
            if abs(dx) > 5 or abs(dy) > 5:
                return 0.0, 0.0
            return float(dx), float(dy)
        except cv2.error:
            return 0.0, 0.0

    def push(self, lr_frame: np.ndarray) -> None:
        """Add one LR frame (H, W) uint8 or float. Updates buffer and shifts."""
        lr = np.asarray(lr_frame, dtype=np.float32)
        if lr.shape[:2] != self.lr_shape:
            return
        if self._ref is None:
            self._ref = lr.copy()
            self._buffer.append((lr.copy(), 0.0, 0.0))
            return
        dx, dy = self._estimate_shift(self._ref, lr)
        self._buffer.append((lr.copy(), dx, dy))

    def get_hr(self):
        """
        Fuse buffered frames onto HR grid using subpixel positions.
        Returns (H*scale, W*scale) uint8, or None if too few frames.
        """
        if len(self._buffer) < 2:
            return None
        h_lr, w_lr = self.lr_shape
        h_hr, w_hr = self.hr_shape
        acc = np.zeros(self.hr_shape, dtype=np.float32)
        wgt = np.zeros(self.hr_shape, dtype=np.float32) + 1e-6

        # Build sampling grid for HR: for each HR (i,j), LR coords are (j/scale, i/scale)
        # For frame with shift (dx, dy), we sample LR at (j/scale + dx, i/scale + dy)
        for frame, dx, dy in self._buffer:
            # For each HR pixel (i,j), sample LR at (x,y) = (j/scale + dx, i/scale + dy)
            # cv2.remap needs map_x, map_y shape (h_hr, w_hr)
            xx = (np.arange(w_hr, dtype=np.float32) / self.scale + dx)
            yy = (np.arange(h_hr, dtype=np.float32) / self.scale + dy)
            map_x = np.broadcast_to(xx, (h_hr, w_hr)).copy()
            map_y = np.broadcast_to(yy[:, np.newaxis], (h_hr, w_hr)).copy()
            sampled = cv2.remap(
                frame,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            acc += sampled
            # Weight by validity (optional: downweight near borders)
            wgt += 1.0

        out = (acc / wgt).astype(np.float32)
        out = np.clip(out, 0, 255)
        return out.astype(np.uint8)

    @property
    def frame_count(self):
        return len(self._buffer)
