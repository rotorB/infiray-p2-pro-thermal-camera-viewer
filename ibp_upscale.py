"""
Iterative Back-Projection (IBP) upscale — training-free, image-agnostic.

Refines an upscaled image so that downscaling it back matches the low-res input.
Uses only the current frame; no natural-image priors. Well-suited for thermal
and other non-photo imagery.

Reference: Irani & Peleg, "Improving resolution by image registration" (1991);
variants used in single-image SR to reduce blur without learning.
"""

import cv2
import numpy as np


def ibp_upscale(
    lr: np.ndarray,
    scale: int,
    n_iters: int = 3,
    lam: float = 0.5,
    up_interp: int = cv2.INTER_LANCZOS4,
    down_interp: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Upscale LR image using Lanczos then refine with IBP.

    Args:
        lr: Low-res image (H, W) uint8.
        scale: Integer scale factor (e.g. 3 for 256x192 -> 768x576).
        n_iters: Number of back-projection iterations (2–5 typical).
        lam: Step size for error correction (0.3–1.5; >1 = over-correction, strongest effect).
        up_interp: Interpolation for upscaling.
        down_interp: Interpolation when downscaling HR to compare to LR.

    Returns:
        High-res (H*scale, W*scale) uint8.
    """
    lr = np.asarray(lr, dtype=np.uint8)
    h, w = lr.shape[:2]
    hr_shape = (w * scale, h * scale)

    hr = cv2.resize(lr, hr_shape, interpolation=up_interp)
    hr = hr.astype(np.float32)
    lr_f = lr.astype(np.float32)

    for _ in range(n_iters):
        lr_recon = cv2.resize(hr, (w, h), interpolation=down_interp)
        err = lr_f - lr_recon
        err_hr = cv2.resize(err, hr_shape, interpolation=cv2.INTER_CUBIC)
        hr = hr + lam * err_hr
        hr = np.clip(hr, 0, 255)

    return hr.astype(np.uint8)
