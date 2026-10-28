using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Pairs every headset RGB image with the thermal frame closest in time. Thermal frames are
    /// kept (as °C, never colours) in a short ring buffer with their time and head pose, so the
    /// pair's thermal data is the one captured at the RGB exposure, not just the latest.
    /// Logs pairing and clock diagnostics every 2 s.
    /// </summary>
    public class FramePairing : MonoBehaviour
    {
        public class ThermalSample
        {
            public long TimeNs;
            public Pose HeadPose;
            public readonly float[] Celsius = new float[ThermalCamera.Width * ThermalCamera.Height];
            public float Min;
            public float Max;
        }

        public struct Pair
        {
            public long RgbTimeNs;
            public Pose RgbCameraPose;
            /// <summary>Ring-buffer slot: valid for about <see cref="Capacity"/> thermal frames.</summary>
            public ThermalSample Thermal;
            /// <summary>Thermal time minus RGB time.</summary>
            public float OffsetMs;
        }

        const int Capacity = 16;          // ~0.6 s of thermal frames at 25 fps
        const float MaxOffsetMs = 30;     // more than ~3/4 of a thermal frame: no pair

        /// <summary>
        /// Delay from the thermal sensor's exposure to the frame arriving over USB, subtracted
        /// from the arrival time. Unknown until calibrated (Phase 2); 0 pairs by arrival time.
        /// </summary>
        public float ThermalLatencyMs;

        public bool HasPair { get; private set; }
        public Pair Latest { get; private set; }

        ThermalCamera _thermal;
        HeadsetRgbCamera _rgb;
        readonly ThermalSample[] _ring = new ThermalSample[Capacity];
        int _count;
        int _next;

        // Diagnostics over the current 2 s window.
        float _nextLog;
        int _rgbFrames, _thermalFrames, _pairs, _misses;
        float _sumAbsOffset, _maxAbsOffset;

        public void Init(ThermalCamera thermal, HeadsetRgbCamera rgb)
        {
            _thermal = thermal;
            _rgb = rgb;
            for (int i = 0; i < Capacity; i++) _ring[i] = new ThermalSample();
            _thermal.FrameReady += OnThermal;
            _rgb.FrameReady += OnRgb;
        }

        void OnThermal()
        {
            var s = _ring[_next];
            s.TimeNs = _thermal.TimeNs - (long)(ThermalLatencyMs * 1e6f);
            s.HeadPose = _thermal.HeadPose;
            s.Min = _thermal.Min;
            s.Max = _thermal.Max;
            System.Array.Copy(_thermal.Celsius, s.Celsius, s.Celsius.Length);
            _next = (_next + 1) % Capacity;
            if (_count < Capacity) _count++;
            _thermalFrames++;
        }

        void OnRgb()
        {
            _rgbFrames++;
            ThermalSample best = null;
            long bestAbs = long.MaxValue;
            for (int i = 0; i < _count; i++)
            {
                long abs = System.Math.Abs(_ring[i].TimeNs - _rgb.TimeNs);
                if (abs < bestAbs) { bestAbs = abs; best = _ring[i]; }
            }
            if (best == null || bestAbs * 1e-6f > MaxOffsetMs)
            {
                _misses++;
                return;
            }

            float offsetMs = (best.TimeNs - _rgb.TimeNs) * 1e-6f;
            Latest = new Pair { RgbTimeNs = _rgb.TimeNs, RgbCameraPose = _rgb.CameraPose, Thermal = best, OffsetMs = offsetMs };
            HasPair = true;
            _pairs++;
            _sumAbsOffset += Mathf.Abs(offsetMs);
            _maxAbsOffset = Mathf.Max(_maxAbsOffset, Mathf.Abs(offsetMs));
        }

        void Update()
        {
            if (_thermal == null || Time.unscaledTime < _nextLog) return;
            _nextLog = Time.unscaledTime + 2f;
            if (_rgbFrames == 0 && _thermalFrames == 0) return;

            // Clock check: XrTime now vs CLOCK_MONOTONIC now; ~0 confirms the shared time base.
            double clockDiffMs = OVRPlugin.GetTimeInSeconds() * 1e3 - ThermalCamera.p2pro_now_ns() * 1e-6;

            // Pose check: head pose from OVRPlugin at the RGB time plus the lens offset should
            // match PassthroughCameraAccess's own pose, or the two pose sources disagree.
            string poseCheck = "n/a";
            if (_rgb.Source != null && _rgb.Source.IsPlaying && _rgb.TimeNs != 0)
            {
                var head = ThermalCamera.HeadPoseAt(_rgb.TimeNs);
                var lens = _rgb.Intrinsics.LensOffset;
                var ours = new Pose(head.position + head.rotation * lens.position, head.rotation * lens.rotation);
                var theirs = _rgb.CameraPose;
                poseCheck = $"{Vector3.Distance(ours.position, theirs.position) * 1000:F1} mm / " +
                            $"{Quaternion.Angle(ours.rotation, theirs.rotation):F2} deg";
            }

            Debug.Log($"[P2Pro] pairing: rgb {_rgbFrames / 2f:F0} fps, thermal {_thermalFrames / 2f:F0} fps, " +
                      $"pairs {_pairs}, misses {_misses}, |offset| mean {(_pairs > 0 ? _sumAbsOffset / _pairs : 0):F1} " +
                      $"max {_maxAbsOffset:F1} ms; XrTime-monotonic {clockDiffMs:F1} ms; pose sources differ {poseCheck}");
            _rgbFrames = _thermalFrames = _pairs = _misses = 0;
            _sumAbsOffset = _maxAbsOffset = 0;
        }

        void OnDestroy()
        {
            if (_thermal != null) _thermal.FrameReady -= OnThermal;
            if (_rgb != null) _rgb.FrameReady -= OnRgb;
        }
    }
}
