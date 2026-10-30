"""
RGB-guided refinement of a thermal temperature field.

The RGB camera has far more spatial resolution than the 256×192 bolometer.
A guided filter transfers *edges* from the aligned RGB image onto the
thermal field without copying RGB texture (printed text, wood grain) into
temperature. Regions where the thermal image itself is flat keep the
plain upsampled values.

Guide and src must already be the same size (warp RGB onto the thermal
display, upsample thermal to that size, then call refine_with_rgb_guide).
"""

from __future__ import annotations

import cv2
import numpy as np


def upsample_float(lr: np.ndarray, scale: int) -> np.ndarray:
    """Lanczos upsample of a float HxW field to (H*scale, W*scale)."""
    lr = np.ascontiguousarray(lr, dtype=np.float32)
    h, w = lr.shape[:2]
    return cv2.resize(lr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 4, eps: float = 1e-4) -> np.ndarray:
    """Fast guided filter (He et al.): box-filter linear model in a window."""
    guide = np.ascontiguousarray(guide, dtype=np.float32)
    src = np.ascontiguousarray(src, dtype=np.float32)
    k = max(1, int(radius) * 2 + 1)
    mean_I = cv2.blur(guide, (k, k))
    mean_p = cv2.blur(src, (k, k))
    mean_Ip = cv2.blur(guide * src, (k, k))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.blur(guide * guide, (k, k))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I
    mean_a = cv2.blur(a, (k, k))
    mean_b = cv2.blur(b, (k, k))
    return mean_a * guide + mean_b


def refine_with_rgb_guide(
    temp_hr: np.ndarray,
    rgb_bgr: np.ndarray,
    radius: int = 4,
    eps: float = 4e-4,
) -> np.ndarray:
    """
    Snap thermal edges to aligned RGB edges. `temp_hr` is float °C at display
    size; `rgb_bgr` is the warped RGB frame of the same HxW.

    Mixing weight is the thermal gradient: RGB-only edges (a poster on a
    uniform wall) do not become fake temperature structure.
    """
    src = np.ascontiguousarray(temp_hr, dtype=np.float32)
    if rgb_bgr is None or rgb_bgr.shape[:2] != src.shape[:2]:
        return src
    if rgb_bgr.ndim == 3:
        guide = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    else:
        guide = rgb_bgr
    guide = guide.astype(np.float32) * (1.0 / 255.0)

    refined = guided_filter(guide, src, radius=radius, eps=eps)

    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    tmag = cv2.magnitude(gx, gy)
    med = float(np.median(tmag))
    mad = float(np.median(np.abs(tmag - med)))
    thr = max(3.0 * 1.4826 * mad, 1e-3)
    w = np.clip(tmag / (4.0 * thr), 0.0, 1.0)
    w = cv2.GaussianBlur(w, (0, 0), 1.2)
    return (w * refined + (1.0 - w) * src).astype(np.float32)
