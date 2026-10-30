using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace P2Pro
{
    /// <summary>
    /// Collects Train-mode samples while the target stays still and the user walks around it:
    /// <list type="bullet">
    /// <item>RGB: target corners in the passthrough image, turned into world rays from the camera
    /// pose; all rays per corner are triangulated into the target's 3D corners.</item>
    /// <item>Thermal: target corners in the °C image plus head poses at several candidate
    /// latencies, for <see cref="CalibrationSolver"/>.</item>
    /// </list>
    /// The target must be warmer and brighter than its surroundings, e.g. a monitor or laptop
    /// screen showing a white page.
    /// </summary>
    public class CalibrationCollector : MonoBehaviour
    {
        public const int LatencySteps = 13;
        public const float LatencyStepMs = 10;      // candidates 0..120 ms

        const float ThermalInterval = 0.1f;         // s between thermal detections
        const float RgbInterval = 0.25f;
        const float MaxThermalTurnDegPerS = 60;
        const float MaxRgbTurnDegPerS = 4;          // RGB image/pose pairing is exact only when still
        const float MaxRgbMoveMPerS = 0.05f;
        const float OutlierRayMm = 15;

        public bool Collecting { get; set; }
        public readonly List<CalibrationSolver.Observation> Thermal = new();
        public string ThermalState { get; private set; } = "-";
        public string RgbState { get; private set; } = "-";
        public int RgbViews => _rays.Count;
        public Vector3[] Corners3D { get; } = new Vector3[4];
        public bool HasCorners3D { get; private set; }
        public float TriangulationRmsMm { get; private set; }
        public float ViewSpreadDeg { get; private set; }
        /// <summary>Thermal samples by target size: near (big), mid, far (small).</summary>
        public readonly int[] DistanceBins = new int[3];
        /// <summary>Thermal samples by target position in the image, 3 x 3.</summary>
        public readonly int[] ImageBins = new int[9];

        ThermalCamera _thermal;
        HeadsetRgbCamera _rgb;
        readonly TargetFinder _finder = new();
        readonly List<(Vector3 camera, Ray[] rays)> _rays = new();
        float _nextThermal, _nextRgb;
        long _pendingRgbTime;
        int _pendingRgbFrame = -1;
        bool _readbackInFlight;

        public void Init(ThermalCamera thermal, HeadsetRgbCamera rgb)
        {
            _thermal = thermal;
            _rgb = rgb;
            _thermal.FrameReady += OnThermal;
            _rgb.FrameReady += OnRgb;
        }

        public void Clear()
        {
            Thermal.Clear();
            _rays.Clear();
            HasCorners3D = false;
            TriangulationRmsMm = ViewSpreadDeg = 0;
            System.Array.Clear(DistanceBins, 0, 3);
            System.Array.Clear(ImageBins, 0, 9);
            ThermalState = RgbState = "-";
        }

        static float TurnRateDeg(long timeNs, out float speed)
        {
            var state = OVRPlugin.GetNodePoseStateAtTime(timeNs * 1e-9, OVRPlugin.Node.Head);
            var w = state.AngularVelocity;
            var v = state.Velocity;
            speed = Mathf.Sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            return Mathf.Sqrt(w.x * w.x + w.y * w.y + w.z * w.z) * Mathf.Rad2Deg;
        }

        void OnThermal()
        {
            if (!Collecting || Time.unscaledTime < _nextThermal) return;
            _nextThermal = Time.unscaledTime + ThermalInterval;

            if (TurnRateDeg(_thermal.TimeNs, out _) > MaxThermalTurnDegPerS) { ThermalState = "turning too fast"; return; }
            const int w = ThermalCamera.Width, h = ThermalCamera.Height;
            if (!_finder.FindThermal(_thermal.Celsius, out var quad, out var reason))
            {
                ThermalState = reason;
                return;
            }

            var obs = new CalibrationSolver.Observation { HeadPoses = new Pose[LatencySteps] };
            for (int k = 0; k < 4; k++) obs.Corners[k] = quad[k];
            for (int l = 0; l < LatencySteps; l++)
                obs.HeadPoses[l] = ThermalCamera.HeadPoseAt(_thermal.TimeNs - (long)(l * LatencyStepMs * 1e6f));
            Thermal.Add(obs);

            float areaFraction = quad.Area / (w * h);
            DistanceBins[areaFraction > 0.15f ? 0 : areaFraction > 0.04f ? 1 : 2]++;
            var c = quad.Center;
            ImageBins[Mathf.Clamp((int)(c.y / h * 3), 0, 2) * 3 + Mathf.Clamp((int)(c.x / w * 3), 0, 2)]++;
            ThermalState = $"ok ({areaFraction * 100:F0}% of frame)";
        }

        void OnRgb()
        {
            if (!Collecting || _readbackInFlight || _pendingRgbFrame >= 0 || Time.unscaledTime < _nextRgb) return;
            if (TurnRateDeg(_rgb.TimeNs, out float speed) > MaxRgbTurnDegPerS || speed > MaxRgbMoveMPerS)
            {
                RgbState = "hold still";
                return;
            }
            // The camera texture is filled on the render thread later this frame: read it back
            // next frame, if no newer image has replaced it by then.
            _pendingRgbTime = _rgb.TimeNs;
            _pendingRgbFrame = Time.frameCount;
        }

        void LateUpdate()
        {
            if (_pendingRgbFrame < 0 || Time.frameCount <= _pendingRgbFrame) return;
            _pendingRgbFrame = -1;
            var texture = _rgb.Texture;
            if (texture == null || _rgb.TimeNs != _pendingRgbTime) return;

            long timeNs = _pendingRgbTime;
            var head = ThermalCamera.HeadPoseAt(timeNs);
            var lens = _rgb.Intrinsics.LensOffset;
            var cameraPose = new Pose(head.position + head.rotation * lens.position, head.rotation * lens.rotation);
            _readbackInFlight = true;
            _nextRgb = Time.unscaledTime + RgbInterval;
            AsyncGPUReadback.Request(texture, 0, TextureFormat.RGBA32, request =>
            {
                _readbackInFlight = false;
                if (request.hasError || !Collecting) return;
                DetectRgb(request.GetData<Color32>(), request.width, request.height, cameraPose);
            });
        }

        void DetectRgb(NativeArray<Color32> pixels, int w, int h, Pose cameraPose)
        {
            if (!_finder.FindRgb(pixels, w, h, out var quad, out var reason))
            {
                RgbState = reason;
                return;
            }

            var rays = new Ray[4];
            for (int k = 0; k < 4; k++)
                rays[k] = _rgb.Source.ViewportPointToRay(new Vector2(quad[k].x / w, quad[k].y / h), cameraPose);
            _rays.Add((cameraPose.position, rays));
            RgbState = $"ok ({quad.Area / (w * h) * 100:F0}% of frame)";
            Triangulate();
        }

        // Least-squares intersection of each corner's rays, one round of outlier rejection.
        void Triangulate()
        {
            if (_rays.Count < 2) return;
            var keep = new bool[_rays.Count];
            for (int round = 0; round < 2; round++)
            {
                double sumSq = 0;
                int count = 0;
                for (int k = 0; k < 4; k++)
                {
                    var a = new double[3, 3];
                    var b = new double[3];
                    for (int i = 0; i < _rays.Count; i++)
                    {
                        if (round > 0 && !keep[i]) continue;
                        var r = _rays[i].rays[k];
                        Vector3 d = r.direction.normalized, o = r.origin;
                        for (int row = 0; row < 3; row++)
                        for (int col = 0; col < 3; col++)
                        {
                            double m = (row == col ? 1 : 0) - d[row] * d[col];
                            a[row, col] += m;
                            b[row] += m * o[col];
                        }
                    }
                    if (!Solve3(a, b, out var p)) return;
                    Corners3D[k] = p;
                }
                for (int i = 0; i < _rays.Count; i++)
                {
                    double worst = 0;
                    for (int k = 0; k < 4; k++)
                    {
                        var r = _rays[i].rays[k];
                        double dist = Vector3.Cross(r.direction.normalized, Corners3D[k] - r.origin).magnitude;
                        worst = System.Math.Max(worst, dist);
                        if (round == 0 || keep[i]) { sumSq += dist * dist; count++; }
                    }
                    if (round == 0) keep[i] = worst * 1000 < OutlierRayMm;
                }
                TriangulationRmsMm = (float)(System.Math.Sqrt(sumSq / System.Math.Max(count, 1)) * 1000);
            }

            var center = (Corners3D[0] + Corners3D[1] + Corners3D[2] + Corners3D[3]) / 4;
            float spread = 0;
            for (int i = 0; i < _rays.Count; i++)
            for (int j = i + 1; j < _rays.Count; j++)
                spread = Mathf.Max(spread, Vector3.Angle(center - _rays[i].camera, center - _rays[j].camera));
            ViewSpreadDeg = spread;
            HasCorners3D = true;
        }

        static bool Solve3(double[,] a, double[] b, out Vector3 x)
        {
            double det = a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
                       - a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
                       + a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]);
            x = default;
            if (System.Math.Abs(det) < 1e-12) return false;
            double Det(int col)
            {
                var m = (double[,])a.Clone();
                for (int r = 0; r < 3; r++) m[r, col] = b[r];
                return m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
                     - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
                     + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]);
            }
            x = new Vector3((float)(Det(0) / det), (float)(Det(1) / det), (float)(Det(2) / det));
            return true;
        }

        /// <summary>Target edge lengths in cm (top, right, bottom, left) from the 3D corners.</summary>
        public string TargetSize()
        {
            if (!HasCorners3D) return "-";
            var s = new float[4];
            for (int k = 0; k < 4; k++) s[k] = Vector3.Distance(Corners3D[k], Corners3D[(k + 1) % 4]) * 100;
            return $"{s[0]:F1} x {s[1]:F1} cm (opposite sides {s[2]:F1}, {s[3]:F1})";
        }

        void OnDestroy()
        {
            if (_thermal != null) _thermal.FrameReady -= OnThermal;
            if (_rgb != null) _rgb.FrameReady -= OnRgb;
        }
    }
}
