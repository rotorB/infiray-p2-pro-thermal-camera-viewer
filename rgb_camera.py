"""
RGB camera capture and real-time thermal-to-RGB image alignment.

Provides:
  - find_rgb_camera_index() — detect MacBook / built-in camera index
  - RGBCamera               — threaded cv2.VideoCapture wrapper
  - GradientAligner         — edge/gradient-based alignment (no face): build gradient
                              maps, ECC matching, EMA-smoothed homography.
  - FaceAligner             — face bbox + 5 landmarks (kept as optional).
"""
from __future__ import annotations

import threading
import time
import subprocess
import re

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Camera discovery
# ---------------------------------------------------------------------------

def find_rgb_camera_index(thermal_index: str | None = None) -> int | None:
    """
    Return the cv2 index of the first *non-thermal* video device.

    On macOS, queries AVFoundation via ffmpeg; falls back to probing cv2 indices.
    """
    import shutil
    ffmpeg_path = shutil.which('ffmpeg') or 'ffmpeg'

    thermal_patterns = re.compile(
        r'USB Camera|PureThermal|P2 Pro|InfiRay', re.IGNORECASE
    )

    try:
        cmd = [ffmpeg_path, '-f', 'avfoundation', '-list_devices', 'true', '-i', '""']
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, timeout=5)
        video_section = True
        for line in result.stderr.split('\n'):
            if 'AVFoundation video devices:' in line:
                video_section = True
                continue
            if 'AVFoundation audio devices:' in line:
                break
            m = re.search(r'\[(\d+)\]\s+(.+)', line)
            if m and video_section:
                idx_str, name = m.group(1), m.group(2)
                if thermal_index is not None and idx_str == str(thermal_index):
                    continue
                if thermal_patterns.search(name):
                    continue
                return int(idx_str)
    except Exception:
        pass

    # Fallback: probe cv2 indices directly
    for idx in range(5):
        if thermal_index is not None and idx == int(thermal_index):
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.release()
            return idx

    return None


# ---------------------------------------------------------------------------
# RGB camera (threaded capture)
# ---------------------------------------------------------------------------

class RGBCamera:
    """
    Threaded wrapper around cv2.VideoCapture.

    Usage:
        cam = RGBCamera(index=0)
        cam.start()
        frame = cam.get_frame()   # latest BGR frame or None
        cam.stop()
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self._index = index
        self._width = width
        self._height = height
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> bool:
        """Open camera and start capture thread. Returns True on success."""
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop capture thread and release camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None

    def get_frame(self) -> np.ndarray | None:
        """Return the latest captured BGR frame (thread-safe), or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_running(self) -> bool:
        return self._running

    def _capture_loop(self):
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)


# ---------------------------------------------------------------------------
# Gradient/edge-based alignment (no face)
# ---------------------------------------------------------------------------

def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel X and Y, then magnitude; clip to 0-255 uint8."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag


def _center_crop_to_aspect(bgr: np.ndarray, target_aspect: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop image to center region with target_aspect = width/height. Returns (cropped_bgr, (x,y,w,h))."""
    h, w = bgr.shape[:2]
    current_aspect = w / max(h, 1)
    if current_aspect >= target_aspect:
        # crop width
        new_w = int(h * target_aspect)
        x = (w - new_w) // 2
        crop = bgr[:, x : x + new_w]
        return crop, (x, 0, new_w, h)
    else:
        # crop height
        new_h = int(w / target_aspect)
        y = (h - new_h) // 2
        crop = bgr[y : y + new_h, :]
        return crop, (0, y, w, new_h)


class GradientAligner:
    """
    Align RGB to thermal by matching edges/gradients (no face detection).

    - Center-crops RGB to thermal aspect ratio so proportions match (pixel-to-pixel).
    - Uses full homography from ECC (not only scale+translate) for correct geometry.
    - Zoom: user-adjustable zoom around center (crop edges / magnify).
    """

    WORK_W: int = 256
    WORK_H: int = 192
    ECC_ITERS: int = 30
    ECC_EPS: float = 1e-3
    EMA_H: float = 0.12   # smooth full 3x3 homography to reduce jitter
    DETECT_INTERVAL: int = 2
    ZOOM_MIN: float = 0.5
    ZOOM_MAX: float = 2.0

    def __init__(self):
        self._H = np.eye(3, dtype=np.float64)
        self._tracking = False
        self._frame_count = 0
        self._last_error: str = ''
        self._last_grad_thermal: np.ndarray | None = None
        self._last_grad_rgb: np.ndarray | None = None
        self._crop_rect: tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h) on full RGB
        self._zoom: float = 1.0

    @property
    def is_tracking(self) -> bool:
        return self._tracking

    @property
    def status_text(self) -> str:
        if self._tracking:
            s = np.mean([self._H[0, 0], self._H[1, 1]])
            return f"GRAD  zoom:{self._zoom:.2f}  scale~{s:.2f}  [ ] zoom"
        if self._last_error:
            return f"gradient align: {self._last_error}"
        return "gradient align: first run..."

    def get_zoom(self) -> float:
        return self._zoom

    def set_zoom(self, z: float):
        self._zoom = float(np.clip(z, self.ZOOM_MIN, self.ZOOM_MAX))

    def update(self, rgb_bgr: np.ndarray, thermal_bgr: np.ndarray) -> bool:
        """
        Run gradient-based ECC. Center-crops RGB to thermal aspect for proportion match;
        uses full homography and smooths it with EMA.
        """
        self._frame_count += 1
        if self._frame_count % self.DETECT_INTERVAL != 0:
            return self._tracking

        dw = thermal_bgr.shape[1]
        dh = thermal_bgr.shape[0]
        aspect_thermal = dw / max(dh, 1)
        ww, wh = self.WORK_W, self.WORK_H

        # Center-crop RGB to thermal aspect so proportions match (pixel-to-pixel)
        rgb_crop, self._crop_rect = _center_crop_to_aspect(rgb_bgr, aspect_thermal)
        w_crop, h_crop = rgb_crop.shape[1], rgb_crop.shape[0]
        if w_crop < 10 or h_crop < 10:
            return self._tracking

        th_work = cv2.resize(thermal_bgr, (ww, wh), interpolation=cv2.INTER_LINEAR)
        rgb_work = cv2.resize(rgb_crop, (ww, wh), interpolation=cv2.INTER_LINEAR)

        gray_t = cv2.cvtColor(th_work, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rgb_work, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray_t, gray_t)
        cv2.equalizeHist(gray_r, gray_r)

        grad_t = _gradient_magnitude(gray_t)
        grad_r = _gradient_magnitude(gray_r)
        self._last_grad_thermal = grad_t
        self._last_grad_rgb = grad_r

        H_work = np.eye(3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.ECC_ITERS,
            self.ECC_EPS,
        )
        try:
            _, H_work = cv2.findTransformECC(
                grad_t.astype(np.float32),
                grad_r.astype(np.float32),
                H_work,
                cv2.MOTION_HOMOGRAPHY,
                criteria,
            )
        except cv2.error as e:
            self._last_error = str(e)[:40]
            return self._tracking

        self._last_error = ''
        H_work = np.asarray(H_work, dtype=np.float64)

        # H_work: rgb_work (ww,wh) -> thermal_work (ww,wh). Map to display and to cropped RGB:
        # H_full maps (x,y) in cropped RGB (w_crop,h_crop) -> (x,y) in display (dw,dh).
        S_crop = np.array([
            [ww / w_crop, 0, 0],
            [0, wh / h_crop, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        S_disp = np.array([
            [dw / ww, 0, 0],
            [0, dh / wh, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        H_full = S_disp @ H_work @ S_crop

        if not self._tracking:
            self._H = H_full
            self._tracking = True
        else:
            self._H = (1.0 - self.EMA_H) * self._H + self.EMA_H * H_full
        return self._tracking

    def warp(self, rgb_bgr: np.ndarray, out_size: tuple) -> np.ndarray:
        """Warp RGB to display: center-crop to aspect, then full homography + zoom around center."""
        x, y, w, h = self._crop_rect
        if w < 2 or h < 2:
            return cv2.resize(rgb_bgr, out_size, interpolation=cv2.INTER_LINEAR)
        rgb_crop = rgb_bgr[y : y + h, x : x + w]

        dw, dh = out_size
        cx, cy = dw / 2.0, dh / 2.0
        # Zoom around center: H_zoom = T(cx,cy) @ S(z,z) @ T(-cx,-cy)
        z = self._zoom
        H_zoom = np.array([
            [z, 0, (1 - z) * cx],
            [0, z, (1 - z) * cy],
            [0, 0, 1],
        ], dtype=np.float64)
        H_final = H_zoom @ self._H

        return cv2.warpPerspective(
            rgb_crop, H_final, out_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def get_debug_gradients(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (grad_thermal, grad_rgb) as uint8 (H,W) for debug overlay, or (None, None)."""
        return (self._last_grad_thermal, self._last_grad_rgb)

    def reset(self):
        self._tracking = False
        self._H = np.eye(3, dtype=np.float64)
        self._crop_rect = (0, 0, 0, 0)
        self._last_error = ''
        self._last_grad_thermal = None
        self._last_grad_rgb = None


# ---------------------------------------------------------------------------
# Real-time face-based alignment (optional)
# ---------------------------------------------------------------------------

class FaceAligner:
    """
    Real-time per-frame RGB-to-thermal alignment using face detection.

    How it works
    ------------
    Every N frames (DETECT_INTERVAL):
      1. Detect face bbox in RGB (Haar) and in thermal (skin-temp blob).
      2. From each bbox derive 5 landmark positions (left/right eye, nose, left/right mouth)
         so that each feature maps to the same semantic point.
      3. Compute homography H from RGB 5 pts -> thermal 5 pts (display coords).
      4. Smooth scale separately with strong EMA (EMA_SCALE) to avoid jitter;
         smooth translation with EMA_TRANSL; optionally smooth shear/skew lightly.

    Every frame:
      - warp(rgb_frame, out_size) applies the smoothed H via warpPerspective.
    """

    SKIN_TEMP_MIN: float = 28.0   # °C lower bound for face skin blob
    SKIN_TEMP_MAX: float = 38.0   # °C upper bound
    EMA_SCALE: float      = 0.06  # very strong smoothing for scale (no jump)
    EMA_TRANSL: float     = 0.22  # smoothing for translation
    DETECT_INTERVAL: int  = 3     # run face detection every N frames
    MIN_FACE_AREA: int    = 50    # min blob area in thermal pixel space
    MAX_LOST_FRAMES: int  = 15    # frames without detection before lost
    MAX_SCALE_CHANGE: float = 0.08  # reject scale change > 8% in one step (outlier)

    # Proportional positions inside face bbox for 5 landmarks (x, y in 0..1)
    # left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner
    LANDMARKS_NORM = np.array([
        [0.25, 0.35], [0.75, 0.35], [0.50, 0.50], [0.35, 0.75], [0.65, 0.75],
    ], dtype=np.float64)

    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self._H = np.eye(3, dtype=np.float64)
        self._scale_smooth = 1.0
        self._tx_smooth = 0.0
        self._ty_smooth = 0.0
        self._tracking = False
        self._lost_count = 0
        self._frame_count = 0
        self._last_rgb_rect: tuple | None = None
        self._last_th_rect: tuple | None = None

    # ------------------------------------------------------------------
    @property
    def is_tracking(self) -> bool:
        return self._tracking

    @property
    def status_text(self) -> str:
        if self._tracking:
            return f"TRACKING  scale:{self._scale_smooth:.2f}  dx:{self._tx_smooth:.0f}  dy:{self._ty_smooth:.0f}"
        if self._last_rgb_rect is None and self._last_th_rect is None:
            return "searching face in both cameras..."
        if self._last_rgb_rect is None:
            return "face found in thermal, searching RGB..."
        if self._last_th_rect is None:
            return "face found in RGB, searching thermal..."
        return "lost — searching..."

    # ------------------------------------------------------------------
    def update(
        self,
        rgb_bgr: np.ndarray,
        temp_celsius: np.ndarray,
        display_scale: int = 3,
    ) -> bool:
        """
        Run face detection (every DETECT_INTERVAL frames) and update EMA homography.

        Parameters
        ----------
        rgb_bgr       : BGR frame from RGB camera (any resolution)
        temp_celsius  : float32/64 array (H_th × W_th), raw thermal grid
        display_scale : thermal upscale factor (e.g. 3 for 192→576)

        Returns True if tracking this frame.
        """
        self._frame_count += 1

        # Run detection only every DETECT_INTERVAL frames for speed
        if self._frame_count % self.DETECT_INTERVAL == 0:
            rgb_rect = self._detect_face_rgb(rgb_bgr)
            th_rect = self._detect_face_thermal(temp_celsius)

            self._last_rgb_rect = rgb_rect
            self._last_th_rect = th_rect

            if rgb_rect is not None and th_rect is not None:
                self._lost_count = 0
                H_new = self._compute_H_5point(rgb_rect, th_rect, display_scale)
                if H_new is None:
                    pass  # keep previous H
                else:
                    s_new = (float(H_new[0, 0]) + float(H_new[1, 1])) / 2.0
                    tx_new = float(H_new[0, 2])
                    ty_new = float(H_new[1, 2])

                    if not self._tracking:
                        self._scale_smooth = s_new
                        self._tx_smooth = tx_new
                        self._ty_smooth = ty_new
                        self._tracking = True
                    else:
                        # Reject huge scale jumps (outliers)
                        if abs(s_new - self._scale_smooth) > self._scale_smooth * self.MAX_SCALE_CHANGE:
                            s_new = self._scale_smooth
                        self._scale_smooth += self.EMA_SCALE * (s_new - self._scale_smooth)
                        self._tx_smooth += self.EMA_TRANSL * (tx_new - self._tx_smooth)
                        self._ty_smooth += self.EMA_TRANSL * (ty_new - self._ty_smooth)

                    self._H = np.array([
                        [self._scale_smooth, 0.0, self._tx_smooth],
                        [0.0, self._scale_smooth, self._ty_smooth],
                        [0.0, 0.0, 1.0],
                    ], dtype=np.float64)
            else:
                self._lost_count += 1
                if self._lost_count > self.MAX_LOST_FRAMES:
                    self._tracking = False

        return self._tracking

    # ------------------------------------------------------------------
    def warp(self, rgb_bgr: np.ndarray, out_size: tuple) -> np.ndarray:
        """Warp rgb_bgr into thermal display space using current H."""
        return cv2.warpPerspective(
            rgb_bgr, self._H, out_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # ------------------------------------------------------------------
    def reset(self):
        """Force re-acquisition (clear tracking state)."""
        self._tracking = False
        self._lost_count = 0
        self._last_rgb_rect = None
        self._last_th_rect = None
        self._scale_smooth = 1.0
        self._tx_smooth = 0.0
        self._ty_smooth = 0.0
        self._H = np.eye(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _landmarks_from_rect(x: int, y: int, w: int, h: int) -> np.ndarray:
        """Return 5 landmark (x,y) in image coords from bbox (x,y,w,h)."""
        pts = np.zeros((5, 2), dtype=np.float64)
        for i, (nx, ny) in enumerate(FaceAligner.LANDMARKS_NORM):
            pts[i, 0] = x + nx * w
            pts[i, 1] = y + ny * h
        return pts

    @classmethod
    def _compute_H_5point(
        cls,
        rgb_rect: tuple,
        th_rect: tuple,
        display_scale: int,
    ) -> np.ndarray | None:
        """
        Compute 3×3 homography from 5 landmark pairs: RGB coords -> thermal display coords.
        Each feature (eyes, nose, mouth) is mapped to the same semantic point.
        """
        rx, ry, rw, rh = rgb_rect
        tx, ty, tw, th = th_rect
        if rw < 5 or rh < 5 or tw < 2 or th < 2:
            return None

        rgb_pts = cls._landmarks_from_rect(rx, ry, rw, rh)
        # Thermal landmarks in temp pixel space -> convert to display space
        th_pts_raw = cls._landmarks_from_rect(tx, ty, tw, th)
        th_pts = th_pts_raw * display_scale

        H, mask = cv2.findHomography(
            rgb_pts.astype(np.float32),
            th_pts.astype(np.float32),
            cv2.RANSAC,
            3.0,
        )
        if H is None or (mask is not None and int(mask.sum()) < 4):
            return None
        return np.asarray(H, dtype=np.float64)

    def _detect_face_rgb(self, bgr: np.ndarray) -> tuple | None:
        """
        Return (x, y, w, h) of the largest face in bgr, or None.

        Downscales to ≤640px wide for speed, then maps coords back.
        """
        h_orig, w_orig = bgr.shape[:2]
        det_scale = min(1.0, 640.0 / max(w_orig, 1))
        if det_scale < 1.0:
            small = cv2.resize(
                bgr,
                (int(w_orig * det_scale), int(h_orig * det_scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            small = bgr

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            return None

        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        inv = 1.0 / det_scale
        return (int(fx * inv), int(fy * inv), int(fw * inv), int(fh * inv))

    def _detect_face_thermal(self, temp_celsius: np.ndarray) -> tuple | None:
        """
        Return (x, y, w, h) of the largest skin-temperature blob, or None.

        Works in temp_celsius pixel space (e.g. 192×256).
        """
        mask = (
            (temp_celsius >= self.SKIN_TEMP_MIN) &
            (temp_celsius <= self.SKIN_TEMP_MAX)
        ).astype(np.uint8) * 255

        # Morphological cleanup: remove noise, fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.MIN_FACE_AREA:
            return None

        return cv2.boundingRect(largest)


# Keep the ORB-based aligner around as an optional manual fallback
class CameraAligner:
    """
    One-shot ORB+RANSAC → ECC alignment (original approach).
    Kept as fallback; prefer FaceAligner for live use.
    """

    _ORB_FEATURES = 1000
    _ORB_RATIO     = 0.75
    _RANSAC_THRESH = 5.0
    _ECC_ITERS     = 50
    _ECC_EPS       = 1e-4
    _CANNY_LO      = 40
    _CANNY_HI      = 120

    def __init__(self):
        self._H: np.ndarray | None = None
        self._H_manual: np.ndarray = np.eye(3, dtype=np.float64)
        self._calibrated = False
        self._last_error: str = ''

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def last_error(self) -> str:
        return self._last_error

    def calibrate(self, thermal_bgr: np.ndarray, rgb_bgr: np.ndarray) -> bool:
        th_h, th_w = thermal_bgr.shape[:2]
        th_gray = cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2GRAY)
        rgb_resized = cv2.resize(rgb_bgr, (th_w, th_h), interpolation=cv2.INTER_LINEAR)
        rgb_gray = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY)
        th_eq = cv2.equalizeHist(th_gray)
        rgb_eq = cv2.equalizeHist(rgb_gray)
        th_edges  = cv2.Canny(th_eq,  self._CANNY_LO, self._CANNY_HI)
        rgb_edges = cv2.Canny(rgb_eq, self._CANNY_LO, self._CANNY_HI)
        H_orb = self._orb_homography(th_edges, rgb_edges)
        if H_orb is None:
            H_orb = np.eye(3, dtype=np.float64)
            self._last_error = 'ORB: not enough matches'
        else:
            self._last_error = ''
        self._H = self._ecc_refine(th_edges, rgb_edges, H_orb)
        self._H_manual = np.eye(3, dtype=np.float64)
        self._calibrated = True
        return True

    def warp(self, rgb_bgr: np.ndarray, out_size: tuple) -> np.ndarray:
        if not self._calibrated or self._H is None:
            return cv2.resize(rgb_bgr, out_size, interpolation=cv2.INTER_LINEAR)
        H = self._H_manual @ self._H
        return cv2.warpPerspective(
            rgb_bgr, H, out_size,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

    def reset(self):
        self._H = None
        self._H_manual = np.eye(3, dtype=np.float64)
        self._calibrated = False
        self._last_error = ''

    def adjust(self, dx: float = 0.0, dy: float = 0.0, ds: float = 1.0):
        T = np.array([[ds, 0.0, dx], [0.0, ds, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        self._H_manual = T @ self._H_manual

    def _orb_homography(self, th_edges, rgb_edges):
        orb = cv2.ORB_create(nfeatures=self._ORB_FEATURES)
        kp1, d1 = orb.detectAndCompute(th_edges, None)
        kp2, d2 = orb.detectAndCompute(rgb_edges, None)
        if d1 is None or d2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            raw = bf.knnMatch(d1, d2, k=2)
        except cv2.error:
            return None
        good = [m for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < self._ORB_RATIO * n.distance]
        if len(good) < 4:
            return None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, self._RANSAC_THRESH)
        if H is None or (mask is not None and int(mask.sum()) < 4):
            return None
        return H.astype(np.float64)

    def _ecc_refine(self, th_edges, rgb_edges, H_init):
        th_h, th_w = th_edges.shape[:2]
        rgb_coarse = cv2.warpPerspective(
            rgb_edges, H_init, (th_w, th_h), flags=cv2.INTER_LINEAR
        ).astype(np.float32)
        th_f = th_edges.astype(np.float32)
        warp_init = np.eye(3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self._ECC_ITERS, self._ECC_EPS,
        )
        try:
            _, H_ecc = cv2.findTransformECC(
                th_f, rgb_coarse, warp_init, cv2.MOTION_HOMOGRAPHY, criteria,
            )
            return H_ecc.astype(np.float64) @ H_init
        except cv2.error:
            return H_init
