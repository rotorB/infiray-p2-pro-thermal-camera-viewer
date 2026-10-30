"""
Fixed-Pattern Noise (FPN) compensation for thermal cameras (Infiray P2 Pro).

Microbolometer pixels keep small individual offset errors after factory
calibration, and the errors drift with sensor temperature. On this module the
visible pattern is not just grain: the ROIC also imprints vertical (column)
and weaker horizontal (row) stripes. Both correctors therefore use a
separable model

    offset[y, x] = pixel[y, x] + col[x] + row[y]

and operate on raw 16-bit sensor units (1/64 K) before temperature conversion.

* FlatFieldCalibrator — the principled fix. Aim at a uniform target (wall /
  lens cap). Frames that moved relative to the first are skipped so a pan
  cannot smear the map. The stack median kills temporal noise and shutter
  pops; a fitted plane absorbs a leftover wall gradient; the rest is stored
  as column + row + pixel offset. Subtracting it removes only the fixed
  pattern (the map is zero-mean, so absolute temperatures stay put).

* FPNCorrector — hands-free scene-based fallback. A gated LMS (Scribner /
  Zuo IRLMS flavour):
    - motion is measured by NCC on blurred half-res frames (not phase
      correlation: whitening lets the FPN itself pin the peak at zero);
    - updates run only when the current frame is a *confident* match of the
      last sample AND has shifted by several pixels from every recent sample
      (static scenes, uniform walls, and two-point oscillations do not
      update — that is the ghosting failure of naive running-mean NUC);
    - column/row offsets are the median of a spatial high-pass (stripes are
      1-D, large objects live in the blur);
    - pixel offsets come from registration-based LMS: after aligning two
      accepted frames, the scene cancels and the residual is FPN.
"""

from __future__ import annotations

import cv2
import numpy as np


def _downsample(f: np.ndarray, blur_sigma: float = 2.0) -> np.ndarray:
    """Blur + halve: kills pixel-grained FPN, keeps scene structure for NCC."""
    s = cv2.GaussianBlur(np.ascontiguousarray(f, dtype=np.float32), (0, 0), blur_sigma)
    return cv2.resize(s, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def _ncc_shift(ref_small: np.ndarray, cur_small: np.ndarray, radius: int):
    """Translate `cur` vs `ref` at full-res pixels.

    Returns (dy, dx, peak, on_border). `on_border` means the peak sat on the
    search window edge, so the numeric shift is a lower bound, not a
    correspondence you can register with.
    """
    m = int(radius)
    ref = np.ascontiguousarray(ref_small, dtype=np.float32)
    cur = np.ascontiguousarray(cur_small, dtype=np.float32)
    if ref.shape != cur.shape:
        return 0.0, 0.0, 0.0, True
    h, w = cur.shape
    if h <= 2 * m + 8 or w <= 2 * m + 8:
        return 0.0, 0.0, 0.0, True
    tpl = np.ascontiguousarray(cur[m:-m, m:-m])
    res = cv2.matchTemplate(ref, tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (mx, my) = cv2.minMaxLoc(res)
    if not np.isfinite(peak):
        peak = 0.0
    rh, rw = res.shape[:2]
    on_border = mx <= 1 or my <= 1 or mx >= rw - 2 or my >= rh - 2
    return 2.0 * float(my - m), 2.0 * float(mx - m), float(peak), on_border


def _refine_shift(ref: np.ndarray, cur: np.ndarray, dy0: float, dx0: float,
                  radius: int = 2, blur: float = 1.0):
    """Integer full-res refinement of a coarse half-res estimate.

    `_ncc_shift` works on a halved frame, so every displacement it reports is
    quantised to 2 full-res pixels: an odd pan comes back off by one, the
    scene no longer cancels in the IRLMS difference, and the leftover edge is
    integrated as if it were FPN. That is the ghosting the gates elsewhere in
    this module exist to prevent, arriving through the back door.

    A small full-res search around the coarse peak recovers the exact integer
    shift. `blur` attenuates the (white) FPN so it cannot pin the peak at zero
    displacement; the caller must still re-check the magnitude gate, because a
    residual pattern stronger than the scene can drag the peak back regardless.

    Returns (dy, dx, peak) in full-res pixels; peak is 0.0 if the overlap was
    too small to match, in which case the coarse estimate is returned as-is.
    """
    r = int(radius)
    dy0, dx0 = int(round(dy0)), int(round(dx0))
    a = cv2.GaussianBlur(np.ascontiguousarray(ref, dtype=np.float32), (0, 0), blur)
    b = cv2.GaussianBlur(np.ascontiguousarray(cur, dtype=np.float32), (0, 0), blur)
    if a.shape != b.shape:
        return float(dy0), float(dx0), 0.0
    h, w = b.shape
    # Template bounds chosen so every candidate shift in [d0-r, d0+r] keeps the
    # matching window inside `ref` (cur[y, x] corresponds to ref[y+dy, x+dx]).
    ty0, ty1 = max(0, r - dy0), min(h, h - dy0 - r)
    tx0, tx1 = max(0, r - dx0), min(w, w - dx0 - r)
    if ty1 - ty0 < 32 or tx1 - tx0 < 32:
        return float(dy0), float(dx0), 0.0
    tpl = np.ascontiguousarray(b[ty0:ty1, tx0:tx1])
    region = np.ascontiguousarray(
        a[ty0 + dy0 - r:ty1 + dy0 + r, tx0 + dx0 - r:tx1 + dx0 + r]
    )
    res = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (mx, my) = cv2.minMaxLoc(res)
    if not np.isfinite(peak):
        return float(dy0), float(dx0), 0.0
    return float(dy0 - r + my), float(dx0 - r + mx), float(peak)


def _fit_plane(img: np.ndarray) -> np.ndarray:
    """Least-squares plane z = a + b x + c y. Absorbs a uniform-target gradient."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack(
        [np.ones(h * w, dtype=np.float64), xx.ravel(), yy.ravel()], axis=1
    )
    coeff, _, _, _ = np.linalg.lstsq(a, img.ravel().astype(np.float64), rcond=None)
    return (coeff[0] + coeff[1] * xx + coeff[2] * yy).astype(np.float32)


def _nonuniformity(img: np.ndarray, sigma: float = 8.0) -> float:
    """How much *scene* structure a would-be flat-field target still holds.

    Removing the fitted plane and blurring hard leaves almost nothing on a real
    uniform target: pixel and column FPN are high-frequency and blur away, and
    a wall gradient is the plane. Anything that survives is large-scale
    structure the sensor is looking at, not glued to it.
    """
    r = img - _fit_plane(img)
    return float(np.std(cv2.GaussianBlur(r, (0, 0), sigma)))


def _mad_cap(residual: np.ndarray, k: float = 3.0) -> float:
    mad = float(np.median(np.abs(residual)))
    return k * 1.4826 * mad + 1e-3


def _decompose_offset(residual: np.ndarray):
    """Split a residual image into (pixel, col, row), all zero-mean."""
    col = np.median(residual, axis=0).astype(np.float32)
    col -= float(col.mean())
    row = np.median(residual - col[np.newaxis, :], axis=1).astype(np.float32)
    row -= float(row.mean())
    pixel = residual.astype(np.float32) - col[np.newaxis, :] - row[:, np.newaxis]
    pixel -= float(pixel.mean())
    return pixel, col, row


class FlatFieldCalibrator:
    """
    Multi-frame flat-field (offset) calibration.

    start() -> feed() each raw frame until it returns True -> apply() forever.
    The offset map can be persisted with save()/load().
    """

    FRAMES_DEFAULT = 48      # ~2 s at 25 fps; median of 48 is robust to pops
    OFFSET_LIMIT = 512.0     # +-8 K in raw units: covers dead pixels, stops wild burn-in
    # A uniform wall has no texture, so NCC is a random "shift". Compare the
    # blurred half-res frame to the first one instead: a still (or still-looking
    # wall) stays; a pan across real structure does not.
    _STILL_MAD = 12.0        # raw units on the downsampled frame
    _BAD_MAD_K = 6.0
    _BAD_MIN = 80.0          # raw units; below this is ordinary FPN, not a stuck pixel
    # The reference is simply the first frame fed. If that one lands on a
    # shutter pop or before the user finished aiming, every good frame after it
    # is rejected and progress sticks forever. After this many consecutive
    # rejects, assume the reference -- not the scene -- is the problem.
    _RESEED_AFTER = 40       # ~1.6 s at 25 fps
    # Holding still is not the same as aiming at something uniform, and only
    # the first was ever checked. Calibrating on a textured scene used to
    # "succeed" and bake an inverted ghost of that scene into the map -- which
    # is then saved and reloaded on every launch, smearing detail in every
    # frame. Valid targets measure ~8-14 here, a faintly textured room ~57.
    _UNIFORM_MAX = 30.0

    def __init__(self, shape, frames=FRAMES_DEFAULT):
        self.shape = tuple(shape)
        self.frames = int(frames)
        self._buf = None      # uint16 stack while calibrating
        self._count = 0
        self._ref_small = None
        self._rejects = 0     # consecutive frames rejected by the stillness gate
        self.reject_reason = None  # why the last calibration attempt was refused
        self._offset = None   # float32 zero-mean map, ready to subtract
        self._bad = None      # bool mask of stuck pixels, replaced in apply()

    @property
    def calibrating(self) -> bool:
        return self._buf is not None

    @property
    def ready(self) -> bool:
        return self._offset is not None

    @property
    def progress(self):
        """(frames_fed, frames_needed) of the calibration in progress."""
        return self._count, self.frames

    def start(self):
        self._buf = np.empty((self.frames,) + self.shape, dtype=np.uint16)
        self._count = 0
        self._ref_small = None
        self._rejects = 0
        self.reject_reason = None

    def cancel(self):
        self._buf = None
        self._count = 0
        self._ref_small = None
        self._rejects = 0

    def feed(self, frame: np.ndarray) -> bool:
        """
        Accumulate one raw frame of the uniform target.
        Frames whose blurred content differs from the first (a pan across
        real structure) are skipped; progress stalls until it matches again.
        Returns True when the map is ready.
        """
        if self._buf is None:
            return False
        frame = np.asarray(frame)
        if frame.shape != self.shape:
            return False
        small = _downsample(frame)
        if self._ref_small is None:
            self._ref_small = small
        else:
            delta = small - self._ref_small
            delta -= float(delta.mean())
            if float(np.median(np.abs(delta))) > self._STILL_MAD:
                self._rejects += 1
                if self._rejects >= self._RESEED_AFTER:
                    # Re-aim on the current frame and start the stack over: the
                    # frames banked against the old reference showed a
                    # different view and must not be averaged with these.
                    self._ref_small = small
                    self._rejects = 0
                    self._count = 0
                return False
            self._rejects = 0

        self._buf[self._count] = frame.astype(np.uint16, copy=False)
        self._count += 1
        if self._count < self.frames:
            return False

        stack = self._buf
        self._buf = None
        self._ref_small = None
        self._rejects = 0
        # Median over time: temporal noise and a one-frame shutter pop vanish;
        # a mean would keep both at 1/sqrt(N).
        mean = np.median(stack, axis=0).astype(np.float32)
        # Refuse a target that is not actually uniform. Checked on the median
        # stack rather than a single frame: least noisy, and it costs nothing
        # to be sure before writing a map that persists across launches.
        if _nonuniformity(mean) > self._UNIFORM_MAX:
            self.reject_reason = 'aim at a uniform surface'
            return False

        # A wall is rarely isothermal. The plane is that leftover gradient;
        # everything else is glued to the sensor.
        residual = mean - _fit_plane(mean)
        pixel, col, row = _decompose_offset(residual)
        offset = pixel + col[np.newaxis, :] + row[:, np.newaxis]
        offset -= float(offset.mean())
        np.clip(offset, -self.OFFSET_LIMIT, self.OFFSET_LIMIT, out=offset)
        self._offset = offset
        self._bad = self._bad_mask_from(offset)
        return True

    def _bad_mask_from(self, offset: np.ndarray):
        hp = offset - cv2.GaussianBlur(offset, (0, 0), 2.0)
        cap = max(_mad_cap(hp, self._BAD_MAD_K), self._BAD_MIN)
        return (np.abs(hp) > cap) & (np.abs(offset) > 64.0)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Subtract the offset map. Returns uint16, unchanged if not ready."""
        if self._offset is None:
            return frame
        f = np.asarray(frame, dtype=np.float32)
        if f.shape != self.shape:
            return frame
        # Round rather than truncate (see FPNCorrector.apply).
        out = np.clip(np.rint(f - self._offset), 0, 65535).astype(np.uint16)
        if self._bad is not None and np.any(self._bad):
            med = cv2.medianBlur(out, 3)
            out = np.where(self._bad, med, out)
        return out

    def save(self, path) -> bool:
        if self._offset is None:
            return False
        try:
            np.save(path, self._offset)
            return True
        except OSError:
            return False

    def load(self, path) -> bool:
        try:
            data = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return False
        if data.shape != self.shape or not np.issubdtype(data.dtype, np.floating):
            return False
        if not np.all(np.isfinite(data)):
            return False
        # Re-centre: save() always writes a zero-mean map, but a truncated or
        # hand-edited file could carry a DC term, which would silently shift
        # every absolute temperature the app reports rather than just removing
        # the fixed pattern.
        off = data.astype(np.float32)
        off -= float(off.mean())
        # A map written before the target check existed may hold a ghost of
        # whatever the camera was aimed at. Real FPN is high-frequency in at
        # least one axis and all but vanishes under a hard blur, so the same
        # measure that vets a target vets a stored map -- otherwise a single
        # bad calibration keeps smearing detail on every launch, for good.
        if _nonuniformity(off) > self._UNIFORM_MAX:
            self.reject_reason = 'saved map looks like a scene; recalibrate'
            return False
        self._offset = np.clip(off, -self.OFFSET_LIMIT, self.OFFSET_LIMIT)
        self._bad = self._bad_mask_from(self._offset)
        return True


class FPNCorrector:
    """
    Scene-based FPN estimator. Every frame:

        corrected = frame - pixel - col - row

    The maps integrate only from frames whose content has actually SHIFTED
    by several pixels since EVERY recently accepted sample, with a
    high NCC peak against the most recent one. Only then does real detail
    land on fresh pixels and cancel, while FPN stays put. Weaker gates fail:

    * "does anything change?" — handheld micro-jitter changes every frame
      yet keeps edges on the same pixels; the static-looking scene slowly
      burns into the map;
    * "moved since the last accepted sample?" — coarse tremor oscillates
      between two spots several pixels apart, each hop passes the pairwise
      test, and the two-position scene still burns in;
    * "NCC peak ignored" — a uniform wall has a noisy peak, random "shifts"
      pass, and the wall is absorbed. Peak + texture gates stop that; use
      FlatFieldCalibrator on a wall instead.

    A static or jittering camera therefore never feeds the map — it keeps
    applying the last learned pattern. Pan the camera to converge.
    """

    STRIPE_SIGMA = 6.0     # 2-D blur: objects larger than this leave the stripe residual
    RATE_COL = 0.10        # 1-D maps have few degrees of freedom; they settle in ~10 pans
    RATE_ROW = 0.06
    RATE_PIX = 0.05        # IRLMS weight per accepted pair
    LIM_COL = 192.0        # +-3 K
    LIM_ROW = 128.0        # +-2 K
    LIM_PIX = 128.0        # +-2 K
    MIN_SHIFT_PX = 4.0     # required displacement from every recent sample
    MIN_PEAK = 0.40        # NCC vs last sample; below this the match is not a correspondence
    MIN_TEXTURE = 20.0     # std of the blurred half-res frame, raw units
    REF_HISTORY = 6        # how many accepted samples the oscillation gate checks
    REFINE_RADIUS = 2      # full-res search around the coarse (even) estimate
    _MATCH_BLUR = 2.0
    _SEARCH_RADIUS = 16    # at half-res: +-32 px full-res window (IRLMS needs a real peak)

    def __init__(self, shape):
        self.shape = tuple(shape)
        h, w = self.shape
        self._pixel = np.zeros(self.shape, dtype=np.float32)
        self._col = np.zeros(w, dtype=np.float32)
        self._row = np.zeros(h, dtype=np.float32)
        self._offset = np.zeros(self.shape, dtype=np.float32)
        self._refs = []            # downsampled frames of recent accepted updates
        self._prev_raw = None      # last accepted raw frame, for IRLMS
        self.n_updates = 0

    @property
    def offset_map(self) -> np.ndarray:
        return self._offset

    def reset(self):
        """Drop the learned map (e.g. after toggling off or a reconnect)."""
        h, w = self.shape
        self._pixel = np.zeros(self.shape, dtype=np.float32)
        self._col = np.zeros(w, dtype=np.float32)
        self._row = np.zeros(h, dtype=np.float32)
        self._offset = np.zeros(self.shape, dtype=np.float32)
        self._refs = []
        self._prev_raw = None
        self.n_updates = 0

    def _compose(self):
        pix = self._pixel
        # Fold 1-D structure out of the pixel map so the three terms stay unique.
        self._col += pix.mean(axis=0)
        pix -= pix.mean(axis=0, keepdims=True)
        self._row += pix.mean(axis=1)
        pix -= pix.mean(axis=1, keepdims=True)
        pix -= float(pix.mean())
        self._col -= float(self._col.mean())
        self._row -= float(self._row.mean())
        np.clip(self._col, -self.LIM_COL, self.LIM_COL, out=self._col)
        np.clip(self._row, -self.LIM_ROW, self.LIM_ROW, out=self._row)
        np.clip(pix, -self.LIM_PIX, self.LIM_PIX, out=pix)
        self._offset = pix + self._col[np.newaxis, :] + self._row[:, np.newaxis]
        self._offset -= float(self._offset.mean())

    def _is_same_place(self, ref_small: np.ndarray, small: np.ndarray) -> bool:
        dy, dx, peak, _ = _ncc_shift(ref_small, small, self._SEARCH_RADIUS)
        if peak < self.MIN_PEAK:
            return False
        return (dy * dy + dx * dx) ** 0.5 < self.MIN_SHIFT_PX

    def _should_update(self, small: np.ndarray):
        """Return (dy, dx, peak, on_border) vs last ref, or None to skip."""
        if not self._refs:
            return None
        if float(small.std()) < self.MIN_TEXTURE:
            return None
        dy, dx, peak, on_border = _ncc_shift(
            self._refs[-1], small, self._SEARCH_RADIUS
        )
        if peak < self.MIN_PEAK:
            return None
        if (dy * dy + dx * dx) ** 0.5 < self.MIN_SHIFT_PX:
            return None
        if any(self._is_same_place(r, small) for r in self._refs[:-1]):
            return None
        return dy, dx, peak, on_border

    def _update_stripes(self, corrected: np.ndarray):
        # Blur along one axis only. A blur across x leaves row structure
        # bit-exact, so it cancels completely in the high-pass and cannot
        # contaminate the column estimate -- and symmetrically for rows. The
        # 2-D blur used before left both patterns in both high-passes, so the
        # two 1-D estimators partly chased each other's error. Two 1-D blurs
        # cost the same as one 2-D blur (OpenCV separates it internally).
        hp_col = corrected - cv2.GaussianBlur(corrected, (0, 1), self.STRIPE_SIGMA)
        col_obs = np.median(hp_col, axis=0).astype(np.float32)
        col_obs -= float(col_obs.mean())
        cap = _mad_cap(col_obs)
        self._col += self.RATE_COL * np.clip(col_obs, -cap, cap)

        hp_row = corrected - cv2.GaussianBlur(
            corrected, (1, 0), 0, sigmaY=self.STRIPE_SIGMA
        )
        row_obs = np.median(hp_row, axis=1).astype(np.float32)
        row_obs -= float(row_obs.mean())
        cap = _mad_cap(row_obs)
        self._row += self.RATE_ROW * np.clip(row_obs, -cap, cap)

    def _update_pixels_irlms(self, raw: np.ndarray, dy: float, dx: float):
        """Registration LMS: scene cancels, remaining difference is FPN."""
        if self._prev_raw is None:
            return
        idy, idx = int(round(dy)), int(round(dx))
        if idy == 0 and idx == 0:
            return
        h, w = self.shape
        # raw[y, x] corresponds to prev_raw[y+idy, x+idx] (see _ncc_shift).
        y0 = max(0, -idy)
        x0 = max(0, -idx)
        y1 = min(h, h - idy)
        x1 = min(w, w - idx)
        margin = 2
        y0 += margin
        x0 += margin
        y1 -= margin
        x1 -= margin
        if y1 - y0 < 16 or x1 - x0 < 16:
            return
        # Both frames corrected with the *current* map so err is remaining FPN.
        cur = raw[y0:y1, x0:x1] - self._offset[y0:y1, x0:x1]
        prv = (
            self._prev_raw[y0 + idy:y1 + idy, x0 + idx:x1 + idx]
            - self._offset[y0 + idy:y1 + idy, x0 + idx:x1 + idx]
        )
        err = cur - prv
        cap = _mad_cap(err)
        np.clip(err, -cap, cap, out=err)
        step = self.RATE_PIX * 0.5 * err
        self._pixel[y0:y1, x0:x1] += step
        self._pixel[y0 + idy:y1 + idy, x0 + idx:x1 + idx] -= step

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Correct one raw frame and (if shifted enough) refine the map."""
        f = np.ascontiguousarray(frame, dtype=np.float32)
        if f.shape != self.shape:
            return frame
        corrected = f - self._offset

        small = _downsample(corrected, self._MATCH_BLUR)
        if not self._refs:
            self._refs.append(small)
            self._prev_raw = f.copy()
        else:
            decision = self._should_update(small)
            if decision is not None:
                dy, dx, _peak, on_border = decision
                self._update_stripes(corrected)
                if not on_border:
                    # Stripes need no registration; the pixel LMS does, and at
                    # full-res precision (see _refine_shift).
                    ry, rx, rpeak = _refine_shift(
                        self._prev_raw - self._offset, corrected, dy, dx,
                        self.REFINE_RADIUS,
                    )
                    # MIN_SHIFT_PX is the "is this a new viewpoint?" test and
                    # _should_update already applied it to the coarse estimate;
                    # re-applying it here would throw away good samples whose
                    # true shift rounds just under it. What the refinement has
                    # to prove is only that it is a real correspondence: a
                    # strong peak, and (structurally) a displacement that stays
                    # within REFINE_RADIUS of a coarse estimate already >=
                    # MIN_SHIFT_PX, so residual FPN cannot pin it at zero.
                    if rpeak >= self.MIN_PEAK and (ry or rx):
                        self._update_pixels_irlms(f, ry, rx)
                self._compose()
                self._refs.append(small)
                if len(self._refs) > self.REF_HISTORY:
                    self._refs.pop(0)
                self._prev_raw = f.copy()
                self.n_updates += 1
                corrected = f - self._offset

        # Round, don't truncate: the offset map is fractional, and .astype()
        # rounds toward zero, which would bias every corrected frame down by
        # half a raw unit (~8 mK, a fifth of this sensor's NETD).
        return np.clip(np.rint(corrected, out=corrected), 0, 65535).astype(np.uint16)
