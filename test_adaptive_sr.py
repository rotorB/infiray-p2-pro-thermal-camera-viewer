"""Image evidence, motion safety, and asynchronous lifecycle regression tests."""
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from adaptive_sr import (AdaptiveSR, Reconstruction, Sample, SamplingModel,
                         estimate_translation, phase_diversity, reconstruct, translate)


def scene(shape=(48, 64)):
    rng = np.random.default_rng(3)
    a = cv2.GaussianBlur(rng.normal(0, 3, shape).astype(np.float32), (0, 0), 1.4) + 25
    a[12:28, 12:25] += 4
    a[32:40, 35:52] += 2
    return a


def synthetic_burst(shape=(48, 64), n=16):
    h, w = shape
    y, x = np.mgrid[:h*2, :w*2].astype(np.float32)
    truth = 25 + .5*np.sin(x/12)*np.cos(y/9)
    truth[14:50, 16:54] += 4
    truth[58:82, 28:100] += 1.5*np.cos(2*np.pi*.58*x[58:82,28:100]/2)
    rng = np.random.default_rng(12)
    positions = rng.uniform(-.8, .8, (n, 2)).astype(np.float32)
    positions[-1] = 0
    model = SamplingModel(shape, positions)
    frames = model.forward((truth-25).ravel()).reshape(n, h, w) + 25
    frames += rng.normal(0, .02, frames.shape).astype(np.float32)
    return truth, [Sample(f, tuple(p), i) for i, (f, p) in enumerate(zip(frames, positions))]


class TestSampling(unittest.TestCase):
    def test_phase_coverage_requires_two_dimensions(self):
        self.assertAlmostEqual(phase_diversity([(0, 0)]*8), 0)
        self.assertLess(phase_diversity([(i, 0) for i in range(8)]), 1e-10)
        self.assertLess(phase_diversity([(i/8, i/8) for i in range(8)]), 1e-10)
        self.assertGreater(phase_diversity([(0, 0), (.5, 0), (0, .5), (.5, .5)]), .99)

    def test_forward_adjoint_and_no_wrap(self):
        model = SamplingModel((20, 24), [(0, 0), (.35, -.6)])
        rng = np.random.default_rng(2)
        x = rng.normal(size=40*48).astype(np.float32)
        y = rng.normal(size=(2,20*24)).astype(np.float32)
        a, b = float(np.sum(model.forward(x)*y)), float(np.sum(x*model.adjoint(y)))
        self.assertAlmostEqual(a, b, delta=2e-5)
        impulse = np.zeros((20,24), np.float32); impulse[10,0] = 1
        shifted = translate(impulse, (-1,0))
        self.assertEqual(float(shifted[:, -1].max()), 0)

    def test_translation_direction_and_subpixel_accuracy(self):
        a=scene((96,128))
        for expected in ((.4, -.3), (-.7,.6), (1.2,.3)):
            b=cv2.warpAffine(a, np.float32([[1,0,expected[0]], [0,1,expected[1]]]),
                             (128,96), borderMode=cv2.BORDER_REFLECT101)
            shift=estimate_translation(a,b)
            self.assertIsNotNone(shift)
            np.testing.assert_allclose(shift, expected, atol=.09)

    def test_reconstruction_recovers_fine_detail(self):
        truth, samples = synthetic_burst()
        result = reconstruct(samples, iterations=22)
        self.assertIsNotNone(result)
        base = cv2.resize(samples[-1].frame, (128,96), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(result.confidence, (128,96), interpolation=cv2.INTER_NEAREST)
        output = base + result.detail*mask
        region = np.s_[62:78, 34:94]
        baseline_error=float(np.mean((base[region]-truth[region])**2))
        improved_error=float(np.mean((output[region]-truth[region])**2))
        self.assertLess(improved_error, baseline_error*.97)

    def test_burst_rejects_temporally_changed_patch(self):
        _, samples = synthetic_burst()
        altered=[]
        for sample in samples:
            f=sample.frame.copy()
            if sample.sequence < 12:
                f[18:28,30:40] += 8
            altered.append(Sample(f,sample.position,sample.sequence))
        result=reconstruct(altered)
        self.assertIsNotNone(result)
        self.assertEqual(float(result.confidence[20:26,32:38].max()), 0)


class TestAdaptiveLifecycle(unittest.TestCase):
    def setUp(self):
        self.sr=AdaptiveSR((48,64), scale=2)
        self.addCleanup(self.sr.close)

    def seed_cache(self):
        a=scene()
        for i in range(12):
            self.sr.push(a, now=1+i*.04)
        self.sr._result=Reconstruction(np.full((96,128), .3, np.float32),
            np.ones((48,64),np.float32),a.copy(),self.sr._position,self.sr._sequence,
            self.sr._generation,.02,1.)
        self.sr._age.fill(20)
        return a

    def test_no_shifts_never_claim_sr(self):
        a=scene()
        for i in range(20):
            self.sr.push(a, now=1+i*.04)
        expected=cv2.resize(a,(128,96),interpolation=cv2.INTER_CUBIC)
        np.testing.assert_array_equal(self.sr.get_hr(),expected)
        self.assertIsNone(self.sr._future)
        self.assertIsNone(self.sr.get_hq())

    def test_moving_object_immediately_uses_current_frame(self):
        a=self.seed_cache()
        self.assertGreater(float(np.max(self.sr.get_hr()-cv2.resize(a,(128,96),
                                                               interpolation=cv2.INTER_CUBIC))), .25)
        moving=a.copy(); moving[20:27,32:40]+=5
        self.sr.push(moving, now=1.48)
        output=self.sr.get_hr()
        base=cv2.resize(moving,(128,96),interpolation=cv2.INTER_CUBIC)
        np.testing.assert_array_equal(output[36:58,60:84],base[36:58,60:84])
        self.sr.push(a, now=1.52)
        output=self.sr.get_hr()
        base=cv2.resize(a,(128,96),interpolation=cv2.INTER_CUBIC)
        np.testing.assert_array_equal(output[36:58,60:84],base[36:58,60:84])

    def test_change_since_worker_reference_is_revalidated(self):
        a=self.seed_cache()
        # Simulate a late result computed before the current thermal change.
        changed=a.copy(); changed[20:28,30:40]+=2
        self.sr._current=changed
        self.sr._age.fill(20)
        base=cv2.resize(changed,(128,96),interpolation=cv2.INTER_CUBIC)
        np.testing.assert_array_equal(self.sr.get_hr()[38:58,58:82],base[38:58,58:82])

    def test_jump_and_gap_clear_history(self):
        a=self.seed_cache()
        generation=self.sr._generation
        self.sr.push(np.flip(a,axis=1).copy(),now=1.48)
        self.assertGreater(self.sr._generation,generation)
        self.assertIsNone(self.sr.get_hq())
        generation=self.sr._generation
        self.sr.push(a,now=2.5)
        self.assertGreater(self.sr._generation,generation)
        self.assertEqual(self.sr.frame_count,1)

    def test_invalid_input_never_reuses_cache(self):
        self.seed_cache()
        a=scene(); a[10,10]=np.nan
        self.sr.push(a,now=1.48)
        self.assertIsNone(self.sr.get_hr())

    def test_float_radiometry_not_display_clipped(self):
        a=np.full((48,64),250.0,np.float32)
        self.sr.push(a,now=1)
        np.testing.assert_allclose(self.sr.get_hr(),250,atol=1e-5)

    def test_reset_rejects_inflight_result_and_never_queues(self):
        entered=threading.Event(); release=threading.Event()
        a=scene()
        def delayed(samples,generation,cancel):
            entered.set()
            release.wait(3)
            ref=samples[-1]
            return Reconstruction(np.ones((96,128),np.float32),np.ones((48,64),np.float32),
                                  ref.frame,ref.position,ref.sequence,generation,.02,1)
        try:
            with patch('adaptive_sr.reconstruct',side_effect=delayed) as worker, \
                 patch('adaptive_sr.estimate_translation',return_value=(.19,.31)):
                for i in range(8):
                    self.sr.push(a,now=1+i*.04)
                self.assertTrue(entered.wait(1))
                future=self.sr._future
                self.sr.reset()
                for i in range(8):
                    self.sr.push(a,now=2+i*.04)
                self.assertIs(self.sr._future,future)
                self.assertEqual(worker.call_count,1)
                release.set()
                future.result(timeout=2)
                self.sr._poll()
                self.assertIsNone(self.sr._result)
        finally:
            release.set()


if __name__ == '__main__':
    cv2.setNumThreads(1)
    unittest.main()
