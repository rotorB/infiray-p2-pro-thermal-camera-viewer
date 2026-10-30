"""Tests for float IBP, RGB-guided refine, MC-Kalman TNR, burst SR."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from app import (FIRST_FRAME_TIMEOUT_S, STALL_TIMEOUT_S, FRAME_SIZE, capture_health,
                 _ffmpeg_capture_command, decode_thermal_frame)
from usb_p2pro import format_unavailable_hint
from burst_sr import BurstSR, richardson_lucy
from denoise import ThermalDenoiser
from guided_upsample import guided_filter, refine_with_rgb_guide, upsample_float
from ibp_upscale import ibp_upscale_float


class TestThermalDecode(unittest.TestCase):
    def test_reads_only_bottom_plane_as_little_endian_words(self):
        # The upper image is not radiometry. High bytes in it must not leak
        # into temperature readings, and adjacent thermal pixels stay separate.
        thermal = np.tile(np.array([19616, 19712], dtype='<u2'), (192, 128))
        payload = b'\xff' * (FRAME_SIZE // 2) + thermal.tobytes()
        decoded = decode_thermal_frame(payload)
        np.testing.assert_array_equal(decoded, thermal)
        c = decoded.astype(np.float32) / 64.0 - 273.15
        self.assertAlmostEqual(float(c[0, 0]), 33.35, places=3)
        self.assertAlmostEqual(float(c[0, 1]), 34.85, places=3)
        decoded[:] = 0
        np.testing.assert_array_equal(decode_thermal_frame(payload), thermal)

    def test_rejects_partial_or_oversize_frames(self):
        for size in (0, FRAME_SIZE // 2, FRAME_SIZE - 1, FRAME_SIZE + 1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                decode_thermal_frame(bytes(size))


class TestIbpFloat(unittest.TestCase):
    def test_roundtrip_stays_near_lr(self):
        rng = np.random.RandomState(0)
        lr = rng.randn(48, 64).astype(np.float32) * 3 + 20.0
        hr = ibp_upscale_float(lr, scale=3, n_iters=4, lam=0.6)
        recon = cv2.resize(hr, (64, 48), interpolation=cv2.INTER_AREA)
        self.assertLess(float(np.sqrt(np.mean((recon - lr) ** 2))), 0.35)

    def test_does_not_clip_celsius_range(self):
        lr = np.full((32, 40), 37.5, np.float32)
        hr = ibp_upscale_float(lr, scale=2, n_iters=2)
        self.assertTrue(np.all(np.isfinite(hr)))
        self.assertGreater(float(hr.mean()), 30.0)


class TestGuided(unittest.TestCase):
    def test_upsample_size(self):
        lr = np.zeros((16, 24), np.float32)
        hr = upsample_float(lr, 3)
        self.assertEqual(hr.shape, (48, 72))

    def test_guided_follows_guide_edge(self):
        h, w = 48, 64
        src = np.zeros((h, w), np.float32)
        src[:, w // 2 :] = 10.0
        src = cv2.GaussianBlur(src, (0, 0), 2.5)
        guide = np.zeros((h, w), np.float32)
        guide[:, w // 2 :] = 1.0
        out = guided_filter(guide, src, radius=4, eps=1e-4)
        def edge_width(img):
            col = img[h // 2]
            return float(np.sum(np.abs(np.diff(col)) > 0.3))
        # Guided output should concentrate the step (fewer intermediate columns).
        self.assertLess(edge_width(out), edge_width(src))

    def test_refine_ignores_rgb_texture_on_flat_thermal(self):
        h, w = 40, 56
        src = np.full((h, w), 22.0, np.float32)
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[:, 10:12] = 255  # a vertical "poster edge" with no thermal edge
        out = refine_with_rgb_guide(src, rgb)
        self.assertLess(float(np.max(np.abs(out - src))), 0.15)


class TestKalmanTNR(unittest.TestCase):
    def test_static_noise_drops(self):
        rng = np.random.RandomState(1)
        true = np.full((48, 64), 21.0, np.float32)
        d = ThermalDenoiser(true.shape)
        d.strength = 8.0
        last = None
        for _ in range(40):
            z = true + rng.randn(*true.shape).astype(np.float32) * 0.8
            last = d.apply(z)
        self.assertIsNotNone(last)
        self.assertLess(float(np.std(last)), 0.25)

    def test_static_edge_is_not_washed_out(self):
        rng = np.random.RandomState(2)
        true = np.zeros((48, 64), np.float32)
        true[:, 32:] = 8.0
        d = ThermalDenoiser(true.shape)
        d.strength = 6.0
        out = true.copy()
        for _ in range(25):
            z = true + rng.randn(*true.shape).astype(np.float32) * 0.3
            out = d.apply(z)
        left = float(out[:, :20].mean())
        right = float(out[:, 44:].mean())
        self.assertLess(left, 1.5)
        self.assertGreater(right, 6.5)


class TestBurstSR(unittest.TestCase):
    def test_fuse_shape_and_finite(self):
        h, w = 32, 48
        sr = BurstSR((h, w), scale=3, max_frames=8)
        canvas = np.zeros((h, w + 6), np.float32)
        yy, xx = np.mgrid[0:h, 0:w + 6]
        canvas += 18.0 + 4.0 * np.sin(xx / 5.0) * np.cos(yy / 4.0)
        for t in range(8):
            sr.push(canvas[:, t:t + w])
        hr = sr.get_hr()
        self.assertIsNotNone(hr)
        self.assertEqual(hr.shape, (h * 3, w * 3))
        self.assertTrue(np.all(np.isfinite(hr)))
        hq = sr.get_hq()
        self.assertIsNotNone(hq)
        self.assertEqual(hq.shape, hr.shape)

    def test_rl_preserves_mean(self):
        img = np.full((24, 24), 12.0, np.float32)
        out = richardson_lucy(img, psf_sigma=0.7, n_iter=4)
        self.assertAlmostEqual(float(out.mean()), 12.0, delta=0.2)


class _FakeReader:
    def __init__(self, running=True, age=None):
        self.running = running
        self._age = age

    def seconds_since_frame(self):
        return self._age


class TestUsbHint(unittest.TestCase):
    def test_not_plugged_in(self):
        msg = format_unavailable_hint({'present': False})
        self.assertIn('not on USB', msg)

    def test_webusb_holders(self):
        msg = format_unavailable_hint({
            'present': True,
            'has_interfaces': False,
            'holders': ['Google Chrome', 'Brave Browser'],
            'composite_unmatched': True,
        })
        self.assertIn('Chrome', msg)
        self.assertIn('WebUSB', msg)

    def test_uvc_unbound_without_holders(self):
        msg = format_unavailable_hint({
            'present': True,
            'has_interfaces': False,
            'holders': [],
            'composite_unmatched': True,
        })
        self.assertIn('UVC', msg)


class TestCaptureHealth(unittest.TestCase):
    def test_no_reader_is_disconnected(self):
        self.assertEqual(capture_health(None, 1.0, now=10.0), "disconnected")

    def test_dead_reader_is_disconnected(self):
        r = _FakeReader(running=False, age=0.1)
        self.assertEqual(capture_health(r, 1.0, now=10.0), "disconnected")

    def test_fresh_open_without_frames_is_not_stalled(self):
        r = _FakeReader(running=True, age=None)
        # 3 s after open, still waiting for the first UVC frame — must NOT kill.
        self.assertIsNone(capture_health(r, capture_started_at=10.0, now=13.0))
        self.assertLess(3.0, FIRST_FRAME_TIMEOUT_S)

    def test_open_with_no_frames_eventually_fails(self):
        r = _FakeReader(running=True, age=None)
        started = 10.0
        now = started + FIRST_FRAME_TIMEOUT_S + 0.1
        self.assertEqual(
            capture_health(r, started, now=now),
            "opened but no frames",
        )

    def test_live_stream_stalls_on_short_gap(self):
        r = _FakeReader(running=True, age=STALL_TIMEOUT_S + 0.2)
        self.assertEqual(
            capture_health(r, capture_started_at=1.0, now=20.0),
            "stalled (no frames)",
        )

    def test_live_stream_ok_when_frames_flow(self):
        r = _FakeReader(running=True, age=0.2)
        self.assertIsNone(capture_health(r, capture_started_at=1.0, now=20.0))

    def test_ffmpeg_skips_audio_and_sets_pixel_format(self):
        cmd = _ffmpeg_capture_command("2")
        self.assertIn("-pixel_format", cmd)
        self.assertEqual(cmd[cmd.index("-i") + 1], "2:none")


if __name__ == '__main__':
    unittest.main()
