"""Conservative, asynchronous 2x thermal burst reconstruction.

The UI always starts with the newest frame. Only a spatial detail correction
from a validated burst is reused, and every displayed frame gates it again.
No neural priors, synchronous solver, or queue of old reconstruction jobs.
The effective PSF is an approximation, not a calibration of the P2 Pro.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np


def noise_sigma(frame):
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    residual = cv2.filter2D(frame, -1, kernel)[2:-2, 2:-2]
    return max(1.0 / 64.0, float(np.median(np.abs(residual))) * 1.4826 / 6.0)


def translate(frame, shift, interpolation=cv2.INTER_LINEAR):
    """Positive shift moves scene content right/down. Never reflect old content."""
    dx, dy = shift
    h, w = frame.shape
    return cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                          flags=interpolation, borderMode=cv2.BORDER_CONSTANT)


def matching_mask(current, prediction, sigma, valid=None):
    """Hard rejection, including a guard around changed edges/occlusions.

The gradient allowance covers LR interpolation/aliasing, not a brightness
normalization: actual thermal changes must invalidate the cached detail.
"""
    # Test motion on the low-frequency structure. Raw subpixel alias changes
    # are precisely the evidence the SR solver needs, not moving objects.
    a = cv2.GaussianBlur(current, (0, 0), .9)
    b = cv2.GaussianBlur(prediction, (0, 0), .9)
    gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    tolerance = max(.08, 2.5 * sigma) + .20 * (np.abs(gx) + np.abs(gy))
    good = (np.abs(a - b) <= tolerance).astype(np.uint8)
    if valid is not None:
        good &= (valid > .999).astype(np.uint8)
    return cv2.erode(good, np.ones((5, 5), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)


def estimate_translation(reference, current):
    """Coarse phase estimate, bounded ECC refinement; reject ambiguous scenes."""
    a = cv2.GaussianBlur(reference, (0, 0), 1.0)
    b = cv2.GaussianBlur(current, (0, 0), 1.0)
    if min(float(a.std()), float(b.std())) < .08:
        return None
    a = (a - a.mean()) / max(float(a.std()), .01)
    b = (b - b.mean()) / max(float(b.std()), .01)
    h, w = a.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    try:
        (dx, dy), response = cv2.phaseCorrelate(a.copy(), b.copy(), window)
        if not np.isfinite([dx, dy, response]).all() or response < .12:
            return None
        if max(abs(dx), abs(dy)) > 3.0:
            return None
        warp = np.float32([[1, 0, dx], [0, 1, dy]])
        score, warp = cv2.findTransformECC(
            a, b, warp, cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 12, .001),
            None, 3)
        shift = warp[:, 2]
        if score < .90 or not np.isfinite(shift).all() or np.max(np.abs(shift)) > 2.5:
            return None
        return float(shift[0]), float(shift[1])
    except cv2.error:
        return None


def phase_diversity(positions):
    """Conditioning of the four sampling phases needed for 2D 2x SR.

Integer motion, no motion, and motion on a single diagonal fail this test.
"""
    p = np.asarray(positions, dtype=np.float64)
    if len(p) < 4:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    matrix = np.exp(2j * np.pi * np.column_stack((x * 0, x, y, x + y)))
    singular = np.linalg.svd(matrix, compute_uv=False)
    return float(singular[-1] / max(singular[0], 1e-9))


@dataclass(frozen=True)
class Sample:
    frame: np.ndarray
    position: tuple
    sequence: int


@dataclass(frozen=True)
class Reconstruction:
    detail: np.ndarray
    confidence: np.ndarray
    reference: np.ndarray
    position: tuple
    sequence: int
    generation: int
    sigma: float
    elapsed_ms: float


class SamplingModel:
    """Effective blur followed by subpixel sampling, with an exact adjoint.

Pixel centers follow OpenCV resize's half-pixel convention. Invalid borders
are masked, not wrapped into the opposite side of the scene.
"""
    def __init__(self, shape, shifts, psf_sigma=.45):
        self.h, self.w = shape
        self.H, self.W = self.h * 2, self.w * 2
        self.kernel = cv2.getGaussianKernel(7, psf_sigma * 2).astype(np.float32)
        gy, gx = np.mgrid[:self.h, :self.w].astype(np.float32)
        self.maps = []
        self.valid = []
        for dx, dy in shifts:
            mx, my = 2 * (gx - dx) + .5, 2 * (gy - dy) + .5
            ix, iy = np.floor(mx).astype(np.int32), np.floor(my).astype(np.int32)
            ax, ay = mx - ix, my - iy
            valid = (ix >= 3) & (iy >= 3) & (ix < self.W - 4) & (iy < self.H - 4)
            taps = []
            for ox, oy, weight in ((0, 0, (1-ax)*(1-ay)), (1, 0, ax*(1-ay)),
                                   (0, 1, (1-ax)*ay), (1, 1, ax*ay)):
                idx = (np.clip(iy + oy, 0, self.H-1) * self.W
                       + np.clip(ix + ox, 0, self.W-1)).ravel()
                taps.append((idx, weight.astype(np.float32).ravel()))
            self.maps.append(taps)
            self.valid.append(valid.ravel().astype(np.float32))
        self.valid = np.stack(self.valid)

    def blur(self, x):
        return cv2.sepFilter2D(x.reshape(self.H, self.W), -1, self.kernel, self.kernel,
                              borderType=cv2.BORDER_CONSTANT)

    def forward(self, x):
        blurred = self.blur(x).ravel()
        return np.stack([sum(blurred[idx] * wt for idx, wt in taps) for taps in self.maps])

    def adjoint(self, values):
        acc = np.zeros(self.H * self.W, np.float64)
        for vals, taps in zip(values, self.maps):
            for idx, wt in taps:
                acc += np.bincount(idx, weights=vals * wt, minlength=acc.size)
        return self.blur(acc.astype(np.float32)).ravel()


def _weighted_laplacian(x, wx, wy):
    a = x.reshape(wx.shape[0], wy.shape[1])
    out = np.zeros_like(a)
    dx, dy = np.diff(a, axis=1) * wx, np.diff(a, axis=0) * wy
    out[:, :-1] -= dx
    out[:, 1:] += dx
    out[:-1] -= dy
    out[1:] += dy
    return out.ravel()


def reconstruct(samples, generation=0, cancel=None, iterations=18):
    """Worker only: masked, regularized joint inversion of original samples."""
    started = time.perf_counter()
    ref = samples[-1]
    h, w = ref.frame.shape
    shifts = [(s.position[0] - ref.position[0], s.position[1] - ref.position[1])
              for s in samples]
    if phase_diversity(shifts) < .07:
        return None
    sigma = noise_sigma(ref.frame)
    ones = np.ones((h, w), np.float32)
    masks, coverage = [], np.zeros((h, w), np.float32)
    for sample, shift in zip(samples, shifts):
        if cancel is not None and cancel.is_set():
            return None
        predicted = translate(ref.frame, shift)
        valid = translate(ones, shift)
        mask = matching_mask(sample.frame, predicted, sigma * 1.42, valid)
        masks.append(mask.ravel())
        coverage += translate(mask, (-shift[0], -shift[1]))
    confidence = (coverage >= .8 * len(samples)).astype(np.float32)
    # Flat patches have no evidence of added spatial detail. Keep the fresh
    # frame there, rather than exposing amplified high-frequency sensor noise.
    local_range = cv2.dilate(ref.frame, np.ones((5, 5), np.uint8)) - cv2.erode(
        ref.frame, np.ones((5, 5), np.uint8))
    confidence *= np.clip((local_range - max(.25, 8*sigma)) / max(.5, 12*sigma), 0, 1)
    if np.count_nonzero(confidence > .1) < 16:
        return None
    model = SamplingModel((h, w), shifts)
    weights = np.stack(masks) * model.valid
    offset = float(np.median(ref.frame))
    observed = np.stack([(s.frame - offset).ravel() for s in samples])
    base = cv2.resize(ref.frame - offset, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    edge_scale = max(.25, 6*sigma)
    wx = 1 / np.maximum(1, np.abs(np.diff(base, axis=1)) / edge_scale)
    wy = 1 / np.maximum(1, np.abs(np.diff(base, axis=0)) / edge_scale)
    lam, anchor = .16 * len(samples)/16, .025 * len(samples)/16
    rhs = model.adjoint(weights * observed) + anchor * base.ravel()
    def operator(x):
        return (model.adjoint(weights * model.forward(x))
                + lam * _weighted_laplacian(x, wx, wy) + anchor*x)
    def dot(a, b):
        return float(np.einsum('i,i->', a, b, dtype=np.float64))
    x = base.ravel().copy()
    residual = rhs - operator(x)
    direction = residual.copy()
    rr = dot(residual, residual)
    for _ in range(iterations):
        if cancel is not None and cancel.is_set():
            return None
        if rr < 1e-10:
            break
        ad = operator(direction)
        denominator = dot(direction, ad)
        if denominator <= 1e-12:
            break
        step = rr / denominator
        x += step * direction
        residual -= step * ad
        next_rr = dot(residual, residual)
        direction = residual + (next_rr / rr) * direction
        rr = next_rr
    candidate = x.reshape(h*2, w*2) + offset
    baseline = base + offset
    # Keep only additional fine detail; do not replace the current scene with
    # an old low-frequency image or perform global temporal averaging.
    correction = candidate - baseline
    correction -= cv2.GaussianBlur(correction, (0, 0), 1.2)
    cap = cv2.resize(.22 * local_range, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
    correction = np.clip(correction, -cap, cap)
    if not np.isfinite(correction).all():
        return None
    return Reconstruction(correction, confidence, ref.frame, ref.position,
                          ref.sequence, generation, sigma,
                          (time.perf_counter() - started) * 1000)


class AdaptiveSR:
    """One UI owner, one bounded background worker, immutable published result."""
    def __init__(self, lr_shape, scale=3, max_frames=16, min_frames=8,
                 min_interval=.45):
        self.lr_shape = tuple(lr_shape)
        self.scale = int(scale)  # display scale; actual reconstruction is 2x
        self.max_frames = int(max_frames)
        self.min_frames = int(min_frames)
        self.min_interval = float(min_interval)
        self._buffer = deque(maxlen=self.max_frames)
        self._pool = None
        self._future = None
        self._cancel = threading.Event()
        self._generation = 0
        self._sequence = 0
        self._position = (0., 0.)
        self._current = None
        self._age = np.zeros(self.lr_shape, np.float32)
        self._last_time = None
        self._last_job_time = -1e9
        self._result = None
        self._last_job_sequence = -1
        self._closed = False
        self._last_error = None
        self.coverage = 0.
        self.solve_ms = 0.
        self.frontend_ms = 0.
        self.diversity = 0.
        self.status = 'SR collecting'

    @property
    def frame_count(self):
        return len(self._buffer)

    def reset(self):
        self._generation += 1
        self._cancel.set()
        self._cancel = threading.Event()
        self._buffer.clear()
        self._current = None
        self._position = (0., 0.)
        self._age.fill(0)
        self._result = None
        self._last_time = None
        self.coverage = self.diversity = 0.
        self.status = 'SR collecting'
        # Keep the future until it finishes: reset must not queue another job
        # behind a still-running cancelled solver.

    def close(self):
        self.reset()
        self._closed = True
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    def _poll(self):
        if self._future is None or not self._future.done():
            return
        try:
            result = self._future.result()
            if result is not None and result.generation == self._generation:
                self._result = result
                self.solve_ms = result.elapsed_ms
                self._last_error = None
        except Exception as exc:
            self._last_error = f'{type(exc).__name__}: {exc}'
            self._result = None
        finally:
            self._future = None

    def push(self, frame, now=None):
        started = time.perf_counter()
        now = time.monotonic() if now is None else float(now)
        a = np.array(frame, dtype=np.float32, copy=True, order='C')
        if self._closed:
            return
        if a.shape != self.lr_shape or not np.isfinite(a).all():
            self.reset()
            self.status = 'SR invalid frame'
            return
        self._poll()
        if self._last_time is not None and (now - self._last_time > .3 or now <= self._last_time):
            self.reset()
        self._sequence += 1
        if self._current is not None:
            shift = estimate_translation(self._current, a)
            if shift is None:
                self.reset()
                self.status = 'SR hold steady'
            else:
                pos = (self._position[0] + shift[0], self._position[1] + shift[1])
                prediction = translate(self._current, shift)
                valid = translate(np.ones(self.lr_shape, np.float32), shift)
                good = matching_mask(a, prediction, noise_sigma(a)*1.42, valid)
                if float(good.mean()) < .45 or max(abs(pos[0]), abs(pos[1])) > 6:
                    self.reset()
                    self.status = 'SR motion'
                else:
                    self._position = pos
                    self._age = np.minimum(translate(self._age, shift,
                                                     cv2.INTER_NEAREST) + 1, 50) * good
        a.setflags(write=False)
        self._current = a
        self._last_time = now
        self._buffer.append(Sample(a, self._position, self._sequence))
        self.diversity = phase_diversity([s.position for s in self._buffer])
        if (len(self._buffer) >= self.min_frames and self.diversity >= .07
                and self._future is None and now-self._last_job_time >= self.min_interval):
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='thermal-sr')
            self._future = self._pool.submit(reconstruct, tuple(self._buffer),
                                             self._generation, self._cancel)
            self._last_job_time = now
            self._last_job_sequence = self._sequence
        self.frontend_ms = (time.perf_counter() - started) * 1000

    def get_hr(self):
        self._poll()
        if self._current is None:
            return None
        h, w = self.lr_shape
        base = cv2.resize(self._current, (w*self.scale, h*self.scale),
                          interpolation=cv2.INTER_CUBIC)
        self.coverage = 0.
        result = self._result
        if result is None or self._sequence - result.sequence > 75:
            self.status = ('SR unavailable' if self._last_error else
                           'SR working' if self._future is not None else
                           'SR collecting' if self.frame_count < self.min_frames else
                           'SR need microshift')
            return base
        delta = (self._position[0] - result.position[0],
                 self._position[1] - result.position[1])
        prediction = translate(result.reference, delta)
        valid = translate(np.ones(self.lr_shape, np.float32), delta)
        good = matching_mask(self._current, prediction, result.sigma*1.42, valid)
        confidence = translate(result.confidence, delta)
        confidence *= good * np.clip((self._age - 3)/5, 0, 1)
        # Nearest mask followed by erosion prevents interpolation from bleeding
        # old detail into an immediately invalidated pixel or its boundary.
        mask = cv2.resize(confidence, (w*self.scale, h*self.scale),
                          interpolation=cv2.INTER_NEAREST)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        gx = ((np.arange(w*self.scale, dtype=np.float32)+.5) * 2/self.scale
              - .5 - delta[0]*2)
        gy = ((np.arange(h*self.scale, dtype=np.float32)+.5) * 2/self.scale
              - .5 - delta[1]*2)
        detail = cv2.remap(result.detail,
                           np.broadcast_to(gx, base.shape).copy(),
                           np.broadcast_to(gy[:, None], base.shape).copy(),
                           cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        output = base + mask * detail
        self.coverage = float(np.mean(mask > .1))
        self.status = f'SR 2x {self.coverage:.0%}' if self.coverage > .005 else 'SR watching motion'
        return output

    def get_hq(self):
        """A validated cached reconstruction; never block the GUI for a snapshot."""
        image = self.get_hr()
        return image if self.coverage > .005 else None
