"""
Optional neural super-resolution via OpenCV dnn_superres (opencv-contrib).

Loads LapSRN from superres_models/: prefers LapSRN_x3.pb, then LapSRN_x2.pb.
x2 gives 512x384; caller must resize to target 768x576 if needed.
Falls back to None if opencv-contrib is not installed or no model file found.
"""

import os

_sr_impl = None
_sr_scale = None


def _get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def init_superres():
    """Try to load LapSRN (x3 or x2). Safe to call multiple times."""
    global _sr_impl, _sr_scale
    if _sr_impl is not None:
        return _sr_impl, _sr_scale

    try:
        cv2 = __import__("cv2")
        if not hasattr(cv2, "dnn_superres"):
            return None, None
        dnn = cv2.dnn_superres
    except Exception:
        return None, None

    model_dir = os.path.join(_get_script_dir(), "superres_models")
    for model_name, scale in [("LapSRN_x3.pb", 3), ("LapSRN_x2.pb", 2)]:
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isfile(model_path):
            continue
        try:
            sr = dnn.DnnSuperResImpl_create()
            sr.readModel(model_path)
            sr.setModel("lapsrn", scale)
            _sr_impl = sr
            _sr_scale = scale
            return _sr_impl, _sr_scale
        except Exception:
            continue
    return None, None


def is_available():
    impl, _ = init_superres()
    return impl is not None


def upsample(img_uint8):
    """
    Upscale single-channel uint8 (H, W) using LapSRN. Returns uint8 array;
    size is (H*scale, W*scale) where scale is 2 or 3. Caller resizes to
    display size if needed.
    """
    if _sr_impl is None:
        init_superres()
    if _sr_impl is None:
        return None
    try:
        return _sr_impl.upsample(img_uint8)
    except Exception:
        return None
