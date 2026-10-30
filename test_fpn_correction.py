"""Synthetic tests for scene-based and flat-field FPN correction."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from fpn_correction import (
    FPNCorrector, FlatFieldCalibrator, _downsample, _fit_plane, _ncc_shift,
    _nonuniformity, _refine_shift,
)


H, W = 192, 256
BASE = 18000.0  # typical P2 Pro raw around room temp


def _texture_canvas(h, w, seed=0):
    rng = np.random.RandomState(seed)
    canvas = np.zeros((h, w), np.float32)
    for _ in range(50):
        x = int(rng.randint(0, w))
        y = int(rng.randint(0, h))
        r = int(rng.randint(10, 28))
        val = float(rng.uniform(-500, 500))
        cv2.circle(canvas, (x, y), r, val, -1)
    return cv2.GaussianBlur(canvas, (0, 0), 2.5)


def _add_fpn(scene, pixel, col, row):
    return scene + pixel + col[np.newaxis, :] + row[:, np.newaxis]


class TestFlatField(unittest.TestCase):
    def test_recovers_column_and_pixel_fpn(self):
        rng = np.random.RandomState(1)
        pixel = rng.randn(H, W).astype(np.float32) * 25.0
        col = rng.randn(W).astype(np.float32) * 50.0
        row = rng.randn(H).astype(np.float32) * 15.0
        pixel -= pixel.mean()
        col -= col.mean()
        row -= row.mean()
        true = pixel + col[np.newaxis, :] + row[:, np.newaxis]
        true -= true.mean()

        ff = FlatFieldCalibrator((H, W), frames=24)
        ff.start()
        for i in range(24):
            noise = rng.randn(H, W).astype(np.float32) * 4.0
            frame = np.clip(BASE + true + noise, 0, 65535).astype(np.uint16)
            done = ff.feed(frame)
            if i < 23:
                self.assertFalse(done)
        self.assertTrue(done)
        self.assertTrue(ff.ready)

        err = ff._offset - true
        # Median of 24, then plane+decompose: column FPN is the easy part.
        col_err = np.median(err, axis=0)
        self.assertLess(float(np.sqrt(np.mean(col_err ** 2))), 8.0)
        self.assertLess(float(np.sqrt(np.mean(err ** 2))), 12.0)

    def test_plane_is_not_baked_in(self):
        rng = np.random.RandomState(2)
        yy, xx = np.mgrid[0:H, 0:W]
        plane = (0.4 * xx + 0.3 * yy).astype(np.float32)
        plane -= plane.mean()
        col = rng.randn(W).astype(np.float32) * 40.0
        col -= col.mean()
        true_fpn = col[np.newaxis, :].astype(np.float32)

        ff = FlatFieldCalibrator((H, W), frames=16)
        ff.start()
        for _ in range(16):
            frame = np.clip(BASE + plane + true_fpn, 0, 65535).astype(np.uint16)
            ff.feed(frame)
        # The fitted plane should absorb the ramp; leftover is column FPN.
        fitted = _fit_plane((BASE + plane + true_fpn).astype(np.float32))
        self.assertGreater(float(np.corrcoef(plane.ravel(), (fitted - fitted.mean()).ravel())[0, 1]), 0.95)
        self.assertTrue(ff.ready)
        recovered_col = np.median(ff._offset, axis=0)
        self.assertGreater(float(np.corrcoef(col, recovered_col)[0, 1]), 0.95)
        plane_zm = plane - float(plane.mean())
        off_zm = ff._offset - float(ff._offset.mean())
        corr_plane = abs(float(np.corrcoef(plane_zm.ravel(), off_zm.ravel())[0, 1]))
        self.assertLess(corr_plane, 0.25)

    def test_moving_frame_is_skipped(self):
        canvas = _texture_canvas(H + 20, W + 20, seed=3)
        ff = FlatFieldCalibrator((H, W), frames=8)
        ff.start()
        still = np.clip(BASE + canvas[0:H, 0:W], 0, 65535).astype(np.uint16)
        self.assertFalse(ff.feed(still))
        self.assertEqual(ff.progress[0], 1)
        moved = np.clip(BASE + canvas[8:H + 8, 10:W + 10], 0, 65535).astype(np.uint16)
        self.assertFalse(ff.feed(moved))
        self.assertEqual(ff.progress[0], 1)  # skipped
        self.assertFalse(ff.feed(still))
        self.assertEqual(ff.progress[0], 2)

    def test_reseeds_after_a_poisoned_reference(self):
        """A bad first frame must not stall calibration forever.

        The stillness gate compares against the first frame fed. If that one
        lands on a shutter pop or before the user finished aiming, every good
        frame after it is rejected and progress sticks at 1/N with no way out.
        """
        rng = np.random.RandomState(0)
        ff = FlatFieldCalibrator((H, W), frames=16)
        ff.start()
        textured = np.clip(BASE + _texture_canvas(H, W, seed=1) * 4, 0, 65535).astype(np.uint16)
        ff.feed(textured)
        self.assertEqual(ff.progress[0], 1)
        wall = np.clip(BASE + rng.randn(H, W).astype(np.float32) * 3, 0, 65535).astype(np.uint16)
        for _ in range(FlatFieldCalibrator._RESEED_AFTER + 16 + 5):
            if ff.feed(wall):
                break
        self.assertTrue(ff.ready, "calibration never recovered from a bad reference")

    def test_correction_does_not_bias_absolute_temperature(self):
        """The offset map is fractional; truncating it would cool every frame.

        .astype(uint16) rounds toward zero, so a zero-mean map would still
        shift the whole image down by ~0.5 raw units (~8 mK, a fifth of this
        sensor's NETD) — a bias on every absolute temperature the app reports.
        """
        rng = np.random.RandomState(0)
        off = rng.uniform(-3, 3, (H, W)).astype(np.float32)
        off -= off.mean()
        ff = FlatFieldCalibrator((H, W))
        ff._offset = off
        ff._bad = None
        out = ff.apply(np.full((H, W), BASE, np.uint16))
        self.assertLess(abs(float(out.mean()) - BASE), 0.05)

    def test_refuses_a_non_uniform_target(self):
        """Holding still is not the same as aiming at something uniform.

        Calibrating on a textured scene used to succeed silently and bake an
        inverted ghost of that scene into the offset map -- which is then saved
        and reloaded on every launch, smearing detail in every later frame.
        """
        rng = np.random.RandomState(0)
        fpn = rng.randn(H, W).astype(np.float32) * 25.0
        fpn -= fpn.mean()
        target = BASE + _texture_canvas(H, W, seed=1) * 1.5
        ff = FlatFieldCalibrator((H, W), frames=8)
        ff.start()
        done = False
        for _ in range(200):
            frame = np.clip(target + fpn, 0, 65535).astype(np.uint16)
            done = ff.feed(frame)
            if done or not ff.calibrating:
                break
        self.assertFalse(done)
        self.assertFalse(ff.ready, "a textured target must not produce a map")
        self.assertTrue(ff.reject_reason)
        # ...and applying the untrained calibrator is a no-op, not a ghost.
        live = np.clip(BASE + _texture_canvas(H, W, seed=9), 0, 65535).astype(np.uint16)
        np.testing.assert_array_equal(ff.apply(live), live)

    def test_accepts_a_uniform_target_with_strong_fpn(self):
        """The uniformity check must not reject a legitimate calibration."""
        rng = np.random.RandomState(2)
        fpn = (rng.randn(H, W).astype(np.float32) * 40.0
               + rng.randn(W).astype(np.float32)[np.newaxis, :] * 90.0
               + rng.randn(H).astype(np.float32)[:, np.newaxis] * 30.0)
        yy, xx = np.mgrid[0:H, 0:W]
        wall = BASE + (0.4 * xx + 0.3 * yy).astype(np.float32)   # gradient is fine
        ff = FlatFieldCalibrator((H, W), frames=8)
        ff.start()
        for _ in range(8):
            noise = rng.randn(H, W).astype(np.float32) * 3.0
            ff.feed(np.clip(wall + fpn + noise, 0, 65535).astype(np.uint16))
        self.assertTrue(ff.ready, "a uniform wall with strong FPN must calibrate")
        self.assertIsNone(ff.reject_reason)

    def test_uniformity_metric_separates_targets(self):
        rng = np.random.RandomState(3)
        fpn = (rng.randn(H, W).astype(np.float32) * 25.0
               + rng.randn(W).astype(np.float32)[np.newaxis, :] * 50.0)
        wall = np.full((H, W), BASE, np.float32) + fpn
        room = BASE + _texture_canvas(H, W, seed=1) * 0.3 + fpn   # only faint texture
        self.assertLess(_nonuniformity(wall), FlatFieldCalibrator._UNIFORM_MAX)
        self.assertGreater(_nonuniformity(room), FlatFieldCalibrator._UNIFORM_MAX)

    def test_load_refuses_a_ghost_contaminated_map(self):
        """A map written before the target check existed must not be installed.

        Otherwise one bad calibration keeps smearing detail on every launch.
        """
        import os
        import tempfile
        rng = np.random.RandomState(0)
        good = (rng.randn(H, W).astype(np.float32) * 25.0
                + rng.randn(W).astype(np.float32)[np.newaxis, :] * 50.0)
        good -= good.mean()
        ghost = good + _texture_canvas(H, W, seed=1) * 1.5
        ghost -= ghost.mean()
        fd, path = tempfile.mkstemp(suffix='.npy')
        os.close(fd)
        try:
            np.save(path, ghost)
            ff = FlatFieldCalibrator((H, W))
            self.assertFalse(ff.load(path))
            self.assertFalse(ff.ready)
            self.assertTrue(ff.reject_reason)
            np.save(path, good)
            ok = FlatFieldCalibrator((H, W))
            self.assertTrue(ok.load(path), "a legitimate map must still load")
            self.assertTrue(ok.ready)
        finally:
            os.remove(path)

    def test_save_load_roundtrip(self):
        import os
        import tempfile
        rng = np.random.RandomState(10)
        ff = FlatFieldCalibrator((H, W), frames=8)
        ff.start()
        col = rng.randn(W).astype(np.float32) * 30.0
        for _ in range(8):
            frame = np.clip(BASE + col[np.newaxis, :] + np.zeros((H, W), np.float32), 0, 65535).astype(np.uint16)
            ff.feed(frame)
        self.assertTrue(ff.ready)
        fd, path = tempfile.mkstemp(suffix='.npy')
        os.close(fd)
        try:
            self.assertTrue(ff.save(path))
            other = FlatFieldCalibrator((H, W))
            self.assertTrue(other.load(path))
            np.testing.assert_allclose(other._offset, ff._offset, atol=1e-5)
            raw = np.clip(BASE + 10, 0, 65535).astype(np.uint16)
            np.testing.assert_array_equal(other.apply(raw), ff.apply(raw))
        finally:
            os.remove(path)


class TestFPNCorrector(unittest.TestCase):
    def test_static_scene_does_not_burn_in(self):
        scene = BASE + _texture_canvas(H, W, seed=4)
        rng = np.random.RandomState(4)
        col = rng.randn(W).astype(np.float32) * 40.0
        col -= col.mean()
        frame = np.clip(_add_fpn(scene, 0, col, np.zeros(H, np.float32)), 0, 65535).astype(np.uint16)
        c = FPNCorrector((H, W))
        for _ in range(40):
            c.apply(frame)
        self.assertEqual(c.n_updates, 0)
        self.assertLess(float(np.max(np.abs(c.offset_map))), 1.0)

    def test_uniform_wall_does_not_false_update(self):
        rng = np.random.RandomState(5)
        col = rng.randn(W).astype(np.float32) * 40.0
        wall = BASE + col[np.newaxis, :] + rng.randn(H, W).astype(np.float32) * 3.0
        c = FPNCorrector((H, W))
        for i in range(30):
            # Tiny fake "motion": 1px of noise, not a real pan.
            jitter = rng.randn(H, W).astype(np.float32) * 2.0
            frame = np.clip(wall + jitter, 0, 65535).astype(np.uint16)
            c.apply(frame)
        self.assertEqual(c.n_updates, 0)

    def test_panning_learns_column_fpn(self):
        canvas = BASE + _texture_canvas(H + 8, W + 90, seed=6)
        rng = np.random.RandomState(6)
        col = rng.randn(W).astype(np.float32) * 45.0
        col -= col.mean()
        pixel = rng.randn(H, W).astype(np.float32) * 8.0
        pixel -= pixel.mean()
        row = np.zeros(H, np.float32)
        c = FPNCorrector((H, W))
        for t in range(70):
            x0 = t
            scene = canvas[0:H, x0:x0 + W]
            frame = np.clip(_add_fpn(scene, pixel, col, row), 0, 65535).astype(np.uint16)
            c.apply(frame)
        self.assertGreater(c.n_updates, 10)
        rec = c._col
        corr = float(np.corrcoef(col, rec)[0, 1])
        self.assertGreater(corr, 0.75)
        # Residual column error should be well below the injected amplitude.
        self.assertLess(float(np.sqrt(np.mean((rec - col) ** 2))), 20.0)

    def test_oscillation_stops_after_one_update(self):
        canvas = BASE + _texture_canvas(H + 8, W + 20, seed=8)
        a = np.clip(canvas[0:H, 0:W], 0, 65535).astype(np.uint16)
        b = np.clip(canvas[0:H, 8:W + 8], 0, 65535).astype(np.uint16)
        c = FPNCorrector((H, W))
        for frame in (a, b, a, b, a, b, a, b):
            c.apply(frame)
        self.assertEqual(c.n_updates, 1)

    def test_panning_reduces_pixel_fpn(self):
        canvas = BASE + _texture_canvas(H + 8, W + 90, seed=9)
        rng = np.random.RandomState(9)
        pixel = rng.randn(H, W).astype(np.float32) * 20.0
        pixel -= pixel.mean()
        c = FPNCorrector((H, W))
        for t in range(80):
            scene = canvas[0:H, t:t + W]
            frame = np.clip(scene + pixel, 0, 65535).astype(np.uint16)
            c.apply(frame)
        self.assertGreater(c.n_updates, 10)
        rec = c._pixel
        corr = float(np.corrcoef(pixel.ravel(), rec.ravel())[0, 1])
        self.assertGreater(corr, 0.35)
        self.assertLess(float(np.sqrt(np.mean((rec - pixel) ** 2))), float(np.sqrt(np.mean(pixel ** 2))) * 0.85)

    def test_refine_recovers_odd_shifts(self):
        """The half-res NCC quantises to 2 px; odd pans need the refinement.

        A 1 px registration error means the scene does not cancel in the IRLMS
        difference, so its edges are integrated as if they were FPN.
        """
        canvas = _texture_canvas(H + 40, W + 40, seed=7)
        pixel = np.random.RandomState(0).randn(H, W).astype(np.float32) * 20.0
        coarse_exact = refined_exact = 0
        cases = [(dy, dx) for dx in range(4, 14) for dy in (0, 3)]
        for true_dy, true_dx in cases:
            a = BASE + canvas[10:H + 10, 10:W + 10] + pixel
            b = BASE + canvas[10 + true_dy:H + 10 + true_dy,
                              10 + true_dx:W + 10 + true_dx] + pixel
            dy, dx, _peak, _border = _ncc_shift(_downsample(a), _downsample(b), 16)
            coarse_exact += (dy == true_dy and dx == true_dx)
            ry, rx, rpeak = _refine_shift(a, b, dy, dx)
            refined_exact += (ry == true_dy and rx == true_dx)
            self.assertGreater(rpeak, 0.5)
        # Half the shifts here are odd, so the coarse estimate cannot get them.
        self.assertLess(coarse_exact, len(cases))
        self.assertEqual(refined_exact, len(cases))

    def test_row_fpn_does_not_leak_into_column_estimate(self):
        """Blurring along one axis only keeps the two 1-D estimators apart.

        A 2-D blur leaves row structure in the column high-pass, so the column
        estimator partly chases row FPN that the row estimator is already
        removing.
        """
        canvas = BASE + _texture_canvas(H + 80, W + 400, seed=5)
        rng = np.random.RandomState(5)
        row = rng.randn(H).astype(np.float32) * 60.0
        row -= row.mean()
        c = FPNCorrector((H, W))
        prng = np.random.RandomState(6)
        x = y = 20.0
        vx = vy = 0.0
        for _ in range(90):
            vx += prng.uniform(-0.6, 0.6)
            vy += prng.uniform(-0.35, 0.35)
            x = float(np.clip(x + np.clip(vx, -3.5, 3.5), 2, 390))
            y = float(np.clip(y + np.clip(vy, -2.0, 2.0), 2, 70))
            scene = canvas[int(y):int(y) + H, int(x):int(x) + W]
            frame = np.clip(scene + row[:, np.newaxis], 0, 65535).astype(np.uint16)
            c.apply(frame)
        self.assertGreater(c.n_updates, 10)
        # No column FPN was injected, so the column map should stay flat.
        self.assertLess(float(np.sqrt(np.mean(c._col ** 2))), 1.0)

    def test_ncc_reports_known_shift(self):
        canvas = _texture_canvas(H + 20, W + 20, seed=7)
        a = _downsample(canvas[0:H, 0:W])
        b = _downsample(canvas[6:H + 6, 10:W + 10])
        dy, dx, peak, border = _ncc_shift(a, b, radius=16)
        self.assertGreater(peak, 0.7)
        self.assertFalse(border)
        # b is canvas[y+6, x+10]; content of b's centre was at +6,+10 in a.
        self.assertAlmostEqual(dy, 6.0, delta=2.0)
        self.assertAlmostEqual(dx, 10.0, delta=2.0)


if __name__ == '__main__':
    unittest.main()
