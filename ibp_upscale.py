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


def ibp_upscale_float(
    lr: np.ndarray,
    scale: int,
    n_iters: int = 3,
    lam: float = 0.5,
    up_interp: int = cv2.INTER_LANCZOS4,
    down_interp: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Lanczos upsample then IBP refine, staying in the source's numeric range.

    Works on temperature (°C) or any other float field. Does not clip to 0–255.
    A loose range guard around the LR min/max stops IBP from exploding.
    """
    lr_f = np.ascontiguousarray(lr, dtype=np.float32)
    h, w = lr_f.shape[:2]
    hr_wh = (w * scale, h * scale)
    hr = cv2.resize(lr_f, hr_wh, interpolation=up_interp)
    lo = float(lr_f.min())
    hi = float(lr_f.max())
    span = max(hi - lo, 1e-3)
    guard_lo, guard_hi = lo - 0.25 * span, hi + 0.25 * span
    for _ in range(max(0, int(n_iters))):
        lr_recon = cv2.resize(hr, (w, h), interpolation=down_interp)
        err = lr_f - lr_recon
        err_hr = cv2.resize(err, hr_wh, interpolation=cv2.INTER_CUBIC)
        hr = hr + float(lam) * err_hr
        np.clip(hr, guard_lo, guard_hi, out=hr)
    return hr


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
    hr = ibp_upscale_float(
        np.asarray(lr, dtype=np.float32),
        scale,
        n_iters=n_iters,
        lam=lam,
        up_interp=up_interp,
        down_interp=down_interp,
    )
    return np.clip(hr, 0, 255).astype(np.uint8)
