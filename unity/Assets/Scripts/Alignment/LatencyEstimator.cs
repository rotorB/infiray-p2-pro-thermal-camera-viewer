using System;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// How long before its USB arrival a thermal frame was exposed. Between consecutive frames the
    /// image shifts as the head turns; the shift is measured from the temperatures (Lucas-Kanade,
    /// one global translation) and predicted from head poses taken at each candidate latency.
    /// The candidate whose predictions match best wins. Turning speed alone does not tell
    /// latencies apart, changes in it do: shaking the head left and right gives the best data.
    /// Can be switched to a manual value.
    /// </summary>
    public class LatencyEstimator : MonoBehaviour
    {
        public const float StepMs = 10;
        const int Candidates = 21;              // 0 .. 200 ms
        const int W = 128, H = 96;              // working resolution (2 x 2 binning)
        const float MinShiftPx = 1.5f;          // full-resolution pixels; smaller motion says little
        const float MaxShiftPx = 12;
        const float MaxErrorPx = 5;             // per-pair error cap (duplicates, bad matches)
        const float Decay = 0.997f;
        const int MinPairs = 40;
        const float TargetDistance = 2;         // m, for the predicted shift

        ThermalPanel _panel;
        ThermalAlignment _alignment;
        float[] _previous = new float[W * H];
        float[] _current = new float[W * H];
        long _previousTime;
        bool _hasPrevious;
        readonly double[] _error = new double[Candidates];
        double _weight;

        public bool Auto { get; set; } = true;
        public float ManualMs { get; set; } = 60;
        /// <summary>Latency used for projection, ms.</summary>
        public float LatencyMs => Auto ? (Estimated ?? ManualMs) : ManualMs;
        /// <summary>The automatic estimate once there is enough motion, else null.</summary>
        public float? Estimated { get; private set; }
        public int Pairs => (int)_weight;

        public void Init(ThermalPanel panel, ThermalAlignment alignment)
        {
            _panel = panel;
            _alignment = alignment;
            _panel.Thermal.FrameReady += OnThermal;
        }

        public void ResetEstimate()
        {
            Array.Clear(_error, 0, Candidates);
            _weight = 0;
            Estimated = null;
        }

        void OnThermal()
        {
            var thermal = _panel.Thermal;
            Downsample(thermal.Celsius, _current);
            long time = thermal.TimeNs;
            bool usable = _hasPrevious && time - _previousTime < 60_000_000;
            if (usable) Measure(_previousTime, time);
            (_previous, _current) = (_current, _previous);
            _previousTime = time;
            _hasPrevious = true;
        }

        void Measure(long t0, long t1)
        {
            var source = _panel.Rgb.Source;
            if (source == null || !source.IsPlaying || !_alignment.Ready) return;
            var mapping = _alignment;

            // A world point ahead of the RGB camera, and where it lands in both frames per candidate.
            var mid = FrameShift(t0, t1, Candidates / 2, source, mapping);
            if (!mid.HasValue || mid.Value.magnitude < MinShiftPx || mid.Value.magnitude > MaxShiftPx) return;
            if (!Track(out var measured)) return;

            for (int i = 0; i < Candidates; i++) _error[i] *= Decay;
            _weight = _weight * Decay + 1;
            for (int i = 0; i < Candidates; i++)
            {
                var predicted = FrameShift(t0, t1, i, source, mapping);
                float e = predicted.HasValue ? Mathf.Min((predicted.Value - measured).magnitude, MaxErrorPx) : MaxErrorPx;
                _error[i] += e * e;
            }
            if (_weight < MinPairs) return;

            int best = 0;
            for (int i = 1; i < Candidates; i++) if (_error[i] < _error[best]) best = i;
            float offset = 0;
            if (best > 0 && best < Candidates - 1)
            {
                double l = _error[best - 1], c = _error[best], r = _error[best + 1];
                double denom = l - 2 * c + r;
                if (denom > 1e-9) offset = (float)Math.Clamp(0.5 * (l - r) / denom, -0.5, 0.5);
            }
            Estimated = (best + offset) * StepMs;
        }

        // Motion of the image content from frame t0 to frame t1, in thermal pixels, if both
        // exposures happened candidate * StepMs before arrival.
        static Vector2? FrameShift(long t0, long t1, int candidate, Meta.XR.PassthroughCameraAccess source,
            ThermalAlignment mapping)
        {
            long lag = (long)(candidate * StepMs * 1e6f);
            var pose0 = RgbProjection.CameraPoseAt(t0 - lag, source);
            var pose1 = RgbProjection.CameraPoseAt(t1 - lag, source);
            var point = pose0.position + pose0.rotation * Vector3.forward * TargetDistance;
            if (!mapping.RgbToThermalUv(Quaternion.Inverse(pose0.rotation) * (point - pose0.position), source, out var uv0) ||
                !mapping.RgbToThermalUv(Quaternion.Inverse(pose1.rotation) * (point - pose1.position), source, out var uv1))
                return null;
            var shift = uv1 - uv0;
            return new Vector2(shift.x * ThermalCamera.Width, shift.y * ThermalCamera.Height);
        }

        static void Downsample(float[] src, float[] dst)
        {
            const int sw = ThermalCamera.Width;
            for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
            {
                int i = 2 * y * sw + 2 * x;
                dst[y * W + x] = 0.25f * (src[i] + src[i + 1] + src[i + sw] + src[i + sw + 1]);
            }
        }

        // One global translation of _current against _previous (content motion), in
        // full-resolution pixels; false when the scene has too little texture.
        bool Track(out Vector2 shift)
        {
            shift = Vector2.zero;
            float lo = float.MaxValue, hi = float.MinValue;
            foreach (float v in _previous) { lo = Mathf.Min(lo, v); hi = Mathf.Max(hi, v); }
            if (hi - lo < 1) return false;
            float scale = 1 / (hi - lo);

            float dx = 0, dy = 0;
            for (int iter = 0; iter < 5; iter++)
            {
                double gxx = 0, gxy = 0, gyy = 0, bx = 0, by = 0;
                for (int y = 4; y < H - 4; y++)
                for (int x = 4; x < W - 4; x++)
                {
                    float i0 = _previous[y * W + x] * scale;
                    float i1 = Bilinear(_current, x + dx, y + dy) * scale;
                    float gx = (Bilinear(_current, x + dx + 1, y + dy) - Bilinear(_current, x + dx - 1, y + dy)) * 0.5f * scale;
                    float gy = (Bilinear(_current, x + dx, y + dy + 1) - Bilinear(_current, x + dx, y + dy - 1)) * 0.5f * scale;
                    float it = i1 - i0;
                    gxx += gx * gx; gxy += gx * gy; gyy += gy * gy;
                    bx -= gx * it; by -= gy * it;
                }
                double det = gxx * gyy - gxy * gxy;
                double minEigen = 0.5 * (gxx + gyy - Math.Sqrt((gxx - gyy) * (gxx - gyy) + 4 * gxy * gxy));
                if (det < 1e-12 || minEigen < 0.05) return false;
                float ux = (float)((gyy * bx - gxy * by) / det);
                float uy = (float)((gxx * by - gxy * bx) / det);
                dx += ux;
                dy += uy;
                if (Mathf.Abs(dx) > MaxShiftPx || Mathf.Abs(dy) > MaxShiftPx) return false;
                if (ux * ux + uy * uy < 1e-4f) break;
            }
            shift = new Vector2(dx, dy) * 2;
            return true;
        }

        static float Bilinear(float[] img, float x, float y)
        {
            x = Mathf.Clamp(x, 0, W - 1.001f);
            y = Mathf.Clamp(y, 0, H - 1.001f);
            int ix = (int)x, iy = (int)y;
            float fx = x - ix, fy = y - iy;
            int i = iy * W + ix;
            float top = img[i] + (img[i + 1] - img[i]) * fx;
            float bottom = img[i + W] + (img[i + W + 1] - img[i + W]) * fx;
            return top + (bottom - top) * fy;
        }

        void OnDestroy()
        {
            if (_panel != null && _panel.Thermal != null) _panel.Thermal.FrameReady -= OnThermal;
        }
    }
}
