using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using Genesis.RoomScan;
using UnityEngine;
using UnityEngine.Rendering;

namespace P2Pro
{
    /// <summary>
    /// Automatic thermal-to-RGB alignment from edges (Setup -> Edge align), no target and no
    /// marking. While the head is still, each thermal frame is paired with the RGB camera's
    /// edges in 3D (P2ProEdgeAlign.compute: RGB luminance edges on the room depth) and the
    /// thermal image's own edges (temperature gradient ridges, <see cref="ThermalEdges"/>).
    /// Solve starts from the calibration in use (or, without any, from the nominal lens turned by
    /// the mount roll), searches the rotation for the best overlap, then fits the
    /// <see cref="ThermalLens"/> so the RGB edge points land on the thermal edges (ICP, point to
    /// line: rotation, focal lengths, centre, k1, k2; the position stays), then a 16 x 12 grid
    /// of offsets for what the model leaves, out to the corners. The result is scored per zone of
    /// the thermal image (centre, middle, rim) against the calibration in use, and Apply stores
    /// it as the Edge calibration (<see cref="ThermalAlignment.EdgeLens"/>) and switches to it.
    /// While collecting, the calibration in use is scored on the latest frames (<see cref="Live"/>).
    /// </summary>
    public class EdgeAlign : MonoBehaviour
    {
        const int Capacity = 32768;             // GPU points per frame
        const int PointsPerFrame = 2500;
        const int MaxFrames = 40;
        public const int MinFrames = 6;
        const float CaptureInterval = 1;        // seconds
        const long StillWindowNs = 300_000_000;
        const float StillDeg = 0.3f, StillM = 0.005f;
        const float MaxDistance = 5;
        static readonly float[] Gates = { 8, 5, 3, 2, 2, 1.5f };   // association distance per round, px
        static readonly float[] ColdGates = { 16, 10, 6, 4, 3, 2, 1.5f };   // from the nominal lens
        const int LiveFrames = 10;
        const float LiveInterval = 1;
        const double Huber = 1;                 // px
        const float GridSmooth = 20, GridRidge = 0.5f, GridMax = 4;
        const int GridMinPoints = 20;
        const float ScoreGate = 3;

        static readonly int WorldToRgbId = Shader.PropertyToID("_WorldToRgb");
        static readonly int WorldToRgbImageId = Shader.PropertyToID("_WorldToRgbImage");
        static readonly int RgbIntrinsicsId = Shader.PropertyToID("_RgbIntrinsics");
        static readonly int RgbCropId = Shader.PropertyToID("_RgbCrop");
        static readonly int RgbEdgeTexId = Shader.PropertyToID("_RgbEdgeTex");
        static readonly int MaxDistanceId = Shader.PropertyToID("_MaxDistance");
        static readonly int CapacityId = Shader.PropertyToID("_Capacity");
        static readonly int PointsId = Shader.PropertyToID("_Points");
        static readonly int CountId = Shader.PropertyToID("_Count");

        /// <summary>One thermal frame and the RGB edge points seen with it.</summary>
        class Frame
        {
            public Vector3[] Points;            // RGB camera frame at the thermal frame's time
            public ThermalEdges Edges;
        }

        /// <summary>Per zone of the thermal image: points near a thermal edge, median and share within 1 px.</summary>
        public struct Zone
        {
            public int Count;
            public float MedianPx, Within1;
            public override string ToString() => Count == 0 ? "-" : $"{MedianPx:F2} px, {Within1 * 100:F0}% <1px ({Count})";
        }

        public static readonly string[] ZoneNames = { "centre", "middle", "rim" };

        ThermalPanel _panel;
        ThermalAlignment _alignment;
        LatencyEstimator _latency;
        Func<float> _edgeThreshold;
        RgbEdgeMap _rgbEdges;
        ComputeShader _shader;
        int _collect, _reset;
        ComputeBuffer _points, _count;
        readonly List<Frame> _frames = new();
        readonly System.Random _random = new(1);
        bool _active, _busy;
        float _nextCapture, _nextLive, _lastCapture = -1;
        int _liveKind = -1;
        Task<(ThermalLens lens, Zone[] before, Zone[] lensOnly, Zone[] after, string log)> _solve;

        public int Frames => _frames.Count;
        public bool Solving => _solve != null && !_solve.IsCompleted;
        /// <summary>Fitted lens with its grid, waiting for Apply; null when none.</summary>
        public ThermalLens Result { get; private set; }
        public Zone[] Before { get; private set; }
        public Zone[] After { get; private set; }
        public bool Better { get; private set; }
        public string Note { get; private set; } = "";
        /// <summary>The calibration in use, scored on the latest frames; null before any.</summary>
        public Zone[] Live { get; private set; }
        /// <summary>The head was still at the last thermal frame.</summary>
        public bool IsStill { get; private set; }
        /// <summary>Seconds since the last frame was taken, or -1.</summary>
        public float SinceCapture => _lastCapture < 0 ? -1 : Time.unscaledTime - _lastCapture;
        /// <summary>RGB edge points of the last frame taken.</summary>
        public int LastPoints { get; private set; }
        public int LastThermalEdges { get; private set; }
        public string LastSolve { get; private set; } = "";

        public void Init(ThermalPanel panel, ThermalAlignment alignment, LatencyEstimator latency, Func<float> edgeThreshold, RgbEdgeMap rgbEdges)
        {
            _rgbEdges = rgbEdges;
            _panel = panel;
            _alignment = alignment;
            _latency = latency;
            _edgeThreshold = edgeThreshold;
            _shader = Resources.Load<ComputeShader>("P2ProEdgeAlign");
            _collect = _shader.FindKernel("Collect");
            _reset = _shader.FindKernel("Reset");
            _points = new ComputeBuffer(Capacity, 16);
            _count = new ComputeBuffer(1, 4);
            foreach (int k in new[] { _collect, _reset }) _shader.SetBuffer(k, CountId, _count);
            _shader.SetBuffer(_collect, PointsId, _points);
            _shader.SetInt(CapacityId, Capacity);
            _shader.SetFloat(MaxDistanceId, MaxDistance);
            _panel.Thermal.FrameReady += OnThermal;
        }

        public void Begin()
        {
            _active = true;
            Note = "";
        }

        public void End() => _active = false;

        public void ClearFrames()
        {
            _frames.Clear();
            Live = null;
            Note = "Frames cleared.";
        }

        public void Discard()
        {
            Result = null;
            Before = After = null;
            Note = "Result discarded.";
        }

        /// <summary>Stores the fitted lens and grid as the Edge calibration and uses it (saved with the alignment).</summary>
        public void Apply()
        {
            if (Result == null) return;
            _alignment.EdgeLens = Result;
            _alignment.Use = (int)ThermalAlignment.Kind.EdgeGrid;
            _alignment.Save();
            Debug.Log("[P2Pro] edge align applied: " + Result);
            Result = null;
            _nextLive = 0;
            Note = $"APPLIED at {DateTime.Now:HH:mm:ss}: now using Edge + grid. The live score below updates with new frames.";
        }

        // ---- Collecting ----

        void OnThermal()
        {
            if (!_active) return;
            var t = _panel.Thermal;
            IsStill = Still(t.TimeNs);
            if (_busy || Solving || Time.unscaledTime < _nextCapture || !IsStill) return;
            var rgb = _panel.Rgb;
            var depth = DepthCapture.Instance;
            if (rgb.Texture == null || _rgbEdges.Texture == null || depth == null || !DepthCapture.DepthAvailable || depth.DepthTex == null) return;
            if (!RgbProjection.TryCreate(_panel, _alignment, _latency, out var projection)) return;
            _nextCapture = Time.unscaledTime + CaptureInterval;

            var celsius = (float[])t.Celsius.Clone();
            _shader.SetMatrixArray(DepthCapture.ViewID, depth.View);
            _shader.SetMatrixArray(DepthCapture.ProjID, depth.Proj);
            _shader.SetMatrixArray(DepthCapture.ViewInvID, depth.ViewInv);
            _shader.SetMatrixArray(DepthCapture.ProjInvID, depth.ProjInv);
            _shader.SetVector(DepthCapture.ZParamsID, depth.Planes);
            _shader.SetInts(DepthCapture.TexSizeID, depth.DepthTex.width, depth.DepthTex.height);
            _shader.SetTexture(_collect, DepthCapture.DepthTexID, depth.DepthTex);
            _shader.SetMatrix(WorldToRgbId, projection.WorldToRgb);
            _shader.SetMatrix(WorldToRgbImageId, ThermalLive.RgbImagePose(rgb).inverse);
            _shader.SetVector(RgbIntrinsicsId, projection.Intrinsics);
            _shader.SetVector(RgbCropId, projection.Crop);
            _shader.SetTexture(_collect, RgbEdgeTexId, _rgbEdges.Texture);
            _shader.Dispatch(_reset, 1, 1, 1);
            _shader.Dispatch(_collect, (depth.DepthTex.width + 7) / 8, (depth.DepthTex.height + 7) / 8, 1);

            _busy = true;
            float threshold = _edgeThreshold();
            AsyncGPUReadback.Request(_count, rc =>
            {
                int count = rc.hasError || _points == null ? 0 : (int)rc.GetData<uint>()[0];
                if (count <= 0) { _busy = false; return; }
                AsyncGPUReadback.Request(_points, r => Store(r, count, celsius, threshold));
            });
        }

        void Store(AsyncGPUReadbackRequest r, int count, float[] celsius, float threshold)
        {
            _busy = false;
            if (r.hasError || !_active) return;
            var data = r.GetData<Vector4>();
            int n = Mathf.Min(count, Capacity);
            // A random subset, so no part of the image dominates by thread order.
            int keep = Mathf.Min(n, PointsPerFrame);
            var points = new Vector3[keep];
            for (int i = 0; i < keep; i++) points[i] = data[i * n / keep + _random.Next(Mathf.Max(1, n / keep))];
            var frame = new Frame { Points = points, Edges = new ThermalEdges(celsius, threshold) };
            if (frame.Edges.Count < 50) { Note = "Too few thermal edges in view: look at warm or cold objects."; return; }
            if (_frames.Count >= MaxFrames) _frames.RemoveAt(0);
            _frames.Add(frame);
            _lastCapture = Time.unscaledTime;
            LastPoints = keep;
            LastThermalEdges = frame.Edges.Count;
            _nextLive = 0;
        }

        // The head moved less than StillDeg / StillM over the last StillWindowNs.
        static bool Still(long timeNs)
        {
            var a = ThermalCamera.HeadPoseAt(timeNs);
            var b = ThermalCamera.HeadPoseAt(timeNs - StillWindowNs);
            return Quaternion.Angle(a.rotation, b.rotation) < StillDeg && Vector3.Distance(a.position, b.position) < StillM;
        }

        // ---- Solving ----

        /// <summary>
        /// Fits the lens and grid on the frames taken. From the calibration in use (or a lens
        /// fitted to the mapping), or <paramref name="fromScratch"/> from the nominal lens turned
        /// by the mount, so the result owes nothing to the other calibrations. Either way it is
        /// scored against the calibration in use.
        /// </summary>
        public void Solve(bool fromScratch)
        {
            if (Solving) return;
            if (_frames.Count < MinFrames) { Note = $"Need {MinFrames} frames, have {_frames.Count}: hold still on a few more views."; return; }
            var source = _panel.Rgb.Source;
            ThermalLens start, current = null;
            var active = _alignment.Active;
            if (active.Solved) current = active.Clone();
            else if (_alignment.Points.Count > 0 && TryLensFromMapping(source, out var fromMapping, out _)) current = fromMapping;
            bool cold = fromScratch || current == null || _alignment.Effective == ThermalAlignment.Kind.Nominal;
            start = cold ? ThermalLens.Nominal(_alignment.MountRollDeg) : current.Clone();
            start.Grid = Array.Empty<float>();
            start.UseGrid = true;
            var frames = _frames.ToArray();
            Note = cold ? $"Solving from the nominal lens (mount {_alignment.MountRollDeg} deg)..." : "Solving from the calibration in use...";
            _solve = Task.Run(() => Run(frames, start, current, cold));
        }

        void Update()
        {
            // The calibration in use, on the latest frames: refreshed each second and on a switch.
            int kind = (int)_alignment.Effective;
            // Only while calibrating: on the main thread, a few ms each time.
            if (_active && _frames.Count > 0 && (Time.unscaledTime >= _nextLive || kind != _liveKind))
            {
                _nextLive = Time.unscaledTime + LiveInterval;
                _liveKind = kind;
                var source = _panel.Rgb.Source;
                var latest = _frames.GetRange(Math.Max(0, _frames.Count - LiveFrames), Math.Min(LiveFrames, _frames.Count)).ToArray();
                Live = kind < 0 || source == null ? null : Scores(latest, c =>
                    _alignment.RgbToThermalUv(c, source, out var uv) ? new Vector2(uv.x * ThermalCamera.Width, uv.y * ThermalCamera.Height) : (Vector2?)null);
            }

            if (_solve == null || !_solve.IsCompleted) return;
            var task = _solve;
            _solve = null;
            if (task.IsFaulted)
            {
                Note = "Solve failed: " + task.Exception?.GetBaseException().Message;
                Debug.LogError("[P2Pro] edge align: " + task.Exception);
                return;
            }
            var (lens, before, lensOnly, after, log) = task.Result;
            Debug.Log("[P2Pro] edge align: " + log);
            LastSolve = $"{DateTime.Now:HH:mm:ss}, {_frames.Count} frames";
            if (lens == null) { Note = "SOLVE FAILED: " + log; return; }
            Result = lens;
            Before = before;
            After = after;
            // More RGB edge points within 1 px of a thermal edge than with the calibration in use.
            Better = Aligned(after) > 1.02f * Aligned(before);
            var sb = new StringBuilder();
            sb.Append(log);
            sb.Append("\nmedian error across edges (share within 1 px, points):");
            for (int z = 0; z < 3; z++)
                sb.Append($"\n  {ZoneNames[z],-7} now {before[z]}  ->  lens {lensOnly[z]}  ->  lens + grid {after[z]}");
            sb.Append(Better ? "\nBETTER than the calibration in use: press Apply." : "\nNOT better than the calibration in use: Discard, or add frames and Solve again.");
            Note = sb.ToString();
            foreach (string line in sb.ToString().Split('\n')) Debug.Log("[P2Pro] edge align: " + line);
        }

        static float Aligned(Zone[] zones)
        {
            float n = 0;
            foreach (var z in zones) n += z.Count * z.Within1;
            return n;
        }

        // Mean of the zones' medians, weighting the rim like the centre.
        static float Score(Zone[] zones)
        {
            float s = 0;
            int n = 0;
            foreach (var z in zones) if (z.Count > 0) { s += z.MedianPx; n++; }
            return n == 0 ? float.MaxValue : s / n;
        }

        // A first lens from the Align-mode mapping: thermal pixels of a grid, as rays at two
        // distances in the RGB camera frame, solved as point pairs.
        bool TryLensFromMapping(Meta.XR.PassthroughCameraAccess source, out ThermalLens lens, out string why)
        {
            lens = new ThermalLens { MountDeg = _alignment.MountRollDeg };
            why = "";
            foreach (float distance in new[] { 0.8f, 2.5f })
            for (int y = 0; y < 6; y++)
            for (int x = 0; x < 8; x++)
            {
                var uv = new Vector2((x + 0.5f) / 8, (y + 0.5f) / 6);
                if (_alignment.ThermalUvToRgb(uv, distance, source, out var rgb))
                    lens.Pairs.Add(new ThermalLens.Pair { Rgb = rgb, ThermalPx = new Vector2(uv.x * ThermalCamera.Width, uv.y * ThermalCamera.Height) });
            }
            string log = lens.Solve();
            if (lens.Solved) return true;
            why = "No starting lens: calibrate first (Calibrate, or the Points wizard). " + log;
            return false;
        }

        // Background: search, fit, grid, scores. Returns a null lens with the reason when it fails.
        static (ThermalLens, Zone[], Zone[], Zone[], string) Run(Frame[] frames, ThermalLens start, ThermalLens current, bool cold)
        {
            var log = new StringBuilder();
            var p = Params(start);
            float searched = Search(frames, p, cold ? 12 : 3, cold ? 1 : 0.5f, out int found);
            log.Append($"search: turned {searched:F1} deg, {found} points on edges; ");
            var rotation = new[] { true, true, true, false, false, false, false, false, false, false, false, false };
            var free = new[] { true, true, true, false, false, false, true, true, true, true, true, true };
            foreach (float gate in cold ? ColdGates : Gates)
            {
                var assoc = Associate(frames, p, null, gate);
                if (assoc.Count < 200) return (null, null, null, null, log + $"only {assoc.Count} edge matches within {gate} px: look at objects with clear warm or cold outlines.");
                double rms = Lm(assoc, p, gate >= 10 ? rotation : free);
                log.Append($"{gate:F1}px: {assoc.Count} matches, {rms:F2} px; ");
            }

            var lens = start.Clone();
            lens.Rotation = CalibrationSolver.ToQuaternion(p[0], p[1], p[2]);
            lens.Fx = (float)p[6]; lens.Fy = (float)p[7]; lens.Cx = (float)p[8]; lens.Cy = (float)p[9];
            lens.K1 = (float)p[10]; lens.K2 = (float)p[11];
            lens.Grid = Array.Empty<float>();
            lens.Solved = true;
            float turned = Quaternion.Angle(lens.Rotation, start.Rotation);
            if (turned > (cold ? 15 : 5) || lens.Fx < 0.8f * start.Fx || lens.Fx > 1.25f * start.Fx || lens.Fy < 0.8f * start.Fy ||
                lens.Fy > 1.25f * start.Fy || Mathf.Abs(lens.K1) > 1)
                return (null, null, null, null, log + $"not plausible (turned {turned:F1} deg, f {lens.Fx:F0} x {lens.Fy:F0}, k1 {lens.K1:F2}); kept the current alignment.");

            var lensOnly = Scores(frames, Projector(lens));
            FitGrid(frames, lens, log);
            var after = Scores(frames, Projector(lens));
            var before = current != null ? Scores(frames, Projector(current)) : new Zone[3];
            lens.RmsPx = Score(after);
            log.Append($"lens f {lens.Fx:F0} x {lens.Fy:F0}, centre {lens.Cx:F0}, {lens.Cy:F0}, k1 {lens.K1:F3} k2 {lens.K2:F3}");
            return (lens, before, lensOnly, after, log.ToString());
        }

        static double[] Params(ThermalLens lens)
        {
            lens.Rotation.ToAngleAxis(out float angle, out var axis);
            if (angle > 180) angle -= 360;
            var w = axis * (angle * Mathf.Deg2Rad);
            return new double[] { w.x, w.y, w.z, lens.Position.x, lens.Position.y, lens.Position.z,
                lens.Fx, lens.Fy, lens.Cx, lens.Cy, lens.K1, lens.K2 };
        }

        // Model pixel of a point in the RGB camera frame, as ThermalLens.RgbToPixel without the grid.
        static bool Project(double[] p, double[] m, Vector3 c, out double px, out double py)
        {
            px = py = 0;
            double dx = c.x - p[3], dy = c.y - p[4], dz = c.z - p[5];
            double tx = m[0] * dx + m[3] * dy + m[6] * dz;
            double ty = m[1] * dx + m[4] * dy + m[7] * dz;
            double tz = m[2] * dx + m[5] * dy + m[8] * dz;
            if (tz <= 0.05) return false;
            double x = tx / tz, y = ty / tz, r2 = x * x + y * y;
            if (r2 > 1.5) return false;
            double d = 1 + r2 * (p[10] + r2 * p[11]);
            px = p[8] + p[6] * x * d;
            py = p[9] + p[7] * y * d;
            return true;
        }

        /// <summary>An RGB edge point matched to the nearest thermal edge: residual = normal . (pixel - edge).</summary>
        struct Match
        {
            public Vector3 Point;
            public Vector2 Edge, Normal;
        }

        // Each point projected (with the grid when given) and matched to the nearest thermal edge within gate.
        static List<Match> Associate(Frame[] frames, double[] p, ThermalLens grid, float gate)
        {
            CalibrationSolver.Rodrigues(p[0], p[1], p[2], out var m);
            var list = new List<Match>();
            foreach (var f in frames)
            foreach (var c in f.Points)
            {
                if (!Project(p, m, c, out double x, out double y)) continue;
                var px = new Vector2((float)x, (float)y);
                if (grid != null) px += grid.GridOffset(px);
                if (!f.Edges.Nearest(px, out var edge, out var normal) || (px - edge).sqrMagnitude > gate * gate) continue;
                list.Add(new Match { Point = c, Edge = edge, Normal = normal });
            }
            return list;
        }

        // Levenberg-Marquardt on point-to-line residuals with Huber weights; returns the RMS.
        static double Lm(List<Match> matches, double[] p, bool[] free)
        {
            int n = matches.Count, np = p.Length;
            var idx = new List<int>();
            for (int j = 0; j < np; j++) if (free[j]) idx.Add(j);
            int f = idx.Count;
            var r = new double[n];
            var rTry = new double[n];
            var w = new double[n];
            var jac = new double[f][];
            for (int j = 0; j < f; j++) jac[j] = new double[n];
            var trial = new double[np];
            Residuals(matches, p, r);
            for (int i = 0; i < n; i++) w[i] = Math.Abs(r[i]) <= Huber ? 1 : Huber / Math.Abs(r[i]);
            double cost = Cost(r, w), lambda = 1e-3;
            for (int iter = 0; iter < 15; iter++)
            {
                for (int j = 0; j < f; j++)
                {
                    int pj = idx[j];
                    double h = 1e-6 * Math.Max(1, Math.Abs(p[pj]));
                    Array.Copy(p, trial, np);
                    trial[pj] += h;
                    Residuals(matches, trial, rTry);
                    for (int i = 0; i < n; i++) jac[j][i] = (rTry[i] - r[i]) / h;
                }
                var a = new double[f, f];
                var g = new double[f];
                for (int j = 0; j < f; j++)
                {
                    double gj = 0;
                    for (int i = 0; i < n; i++) gj -= w[i] * jac[j][i] * r[i];
                    g[j] = gj;
                    for (int k = 0; k <= j; k++)
                    {
                        double s = 0;
                        for (int i = 0; i < n; i++) s += w[i] * jac[j][i] * jac[k][i];
                        a[j, k] = a[k, j] = s;
                    }
                }
                bool improved = false;
                for (int attempt = 0; attempt < 8 && !improved; attempt++)
                {
                    var aa = new double[f, f];
                    for (int j = 0; j < f; j++)
                    for (int k = 0; k < f; k++)
                        aa[j, k] = a[j, k] + (j == k ? lambda * Math.Max(a[j, j], 1e-9) : 0);
                    var step = CalibrationSolver.SolveLinear(aa, (double[])g.Clone());
                    if (step == null) { lambda *= 10; continue; }
                    Array.Copy(p, trial, np);
                    for (int j = 0; j < f; j++) trial[idx[j]] += step[j];
                    Residuals(matches, trial, rTry);
                    double newCost = Cost(rTry, w);
                    if (newCost < cost)
                    {
                        bool converged = cost - newCost < 1e-8 * cost;
                        Array.Copy(trial, p, np);
                        Array.Copy(rTry, r, n);
                        cost = newCost;
                        lambda = Math.Max(lambda / 3, 1e-9);
                        improved = true;
                        if (converged) iter = 100;
                    }
                    else lambda *= 4;
                }
                if (!improved) break;
            }
            return Math.Sqrt(cost / Math.Max(1, n));
        }

        static double Cost(double[] r, double[] w)
        {
            double s = 0;
            for (int i = 0; i < r.Length; i++) s += w[i] * r[i] * r[i];
            return s;
        }

        static void Residuals(List<Match> matches, double[] p, double[] r)
        {
            CalibrationSolver.Rodrigues(p[0], p[1], p[2], out var m);
            for (int i = 0; i < matches.Count; i++)
            {
                var mt = matches[i];
                if (!Project(p, m, mt.Point, out double x, out double y)) { r[i] = 0; continue; }
                r[i] = mt.Normal.x * (x - mt.Edge.x) + mt.Normal.y * (y - mt.Edge.y);
            }
        }

        // Offsets per cell so the model's pixels move onto the thermal edges across their
        // normals: least squares over all matches (bilinear in the cells), smoothed between
        // neighbouring cells and pulled slightly towards 0; cells without data follow their
        // neighbours.
        static void FitGrid(Frame[] frames, ThermalLens lens, StringBuilder log)
        {
            int gx = ThermalLens.GridX, gy = ThermalLens.GridY, cells = gx * gy, u = cells * 2;
            var p = Params(lens);
            var matches = Associate(frames, p, null, 2);
            CalibrationSolver.Rodrigues(p[0], p[1], p[2], out var m);
            var a = new double[u, u];
            var b = new double[u];
            var perCell = new int[cells];
            var ci = new int[4];
            var cw = new double[4];
            foreach (var mt in matches)
            {
                if (!Project(p, m, mt.Point, out double x, out double y)) continue;
                double r = mt.Normal.x * (x - mt.Edge.x) + mt.Normal.y * (y - mt.Edge.y);
                double w = Math.Abs(r) <= Huber ? 1 : Huber / Math.Abs(r);
                // Bilinear weights of the four cells around the pixel.
                double fx = x / ThermalCamera.Width * gx - 0.5, fy = y / ThermalCamera.Height * gy - 0.5;
                int ix = (int)Math.Floor(fx), iy = (int)Math.Floor(fy);
                double tx = fx - ix, ty = fy - iy;
                for (int k = 0; k < 4; k++)
                {
                    int cx = Math.Clamp(ix + (k & 1), 0, gx - 1), cy = Math.Clamp(iy + (k >> 1), 0, gy - 1);
                    ci[k] = cy * gx + cx;
                    cw[k] = ((k & 1) == 1 ? tx : 1 - tx) * ((k >> 1) == 1 ? ty : 1 - ty);
                }
                perCell[ci[0]]++;
                // Row: n . sum_k cw_k o_k = -r.
                for (int k = 0; k < 4; k++)
                for (int dk = 0; dk < 2; dk++)
                {
                    int row = 2 * ci[k] + dk;
                    double jr = cw[k] * (dk == 0 ? mt.Normal.x : mt.Normal.y);
                    b[row] -= w * jr * r;
                    for (int l = 0; l < 4; l++)
                    for (int dl = 0; dl < 2; dl++)
                        a[row, 2 * ci[l] + dl] += w * jr * cw[l] * (dl == 0 ? mt.Normal.x : mt.Normal.y);
                }
            }
            for (int c = 0; c < cells; c++)
            {
                int x = c % gx, y = c / gx;
                for (int d = 0; d < 2; d++)
                {
                    int i = 2 * c + d;
                    a[i, i] += GridRidge;
                    foreach (var (nx, ny) in new[] { (x + 1, y), (x, y + 1) })
                    {
                        if (nx >= gx || ny >= gy) continue;
                        int j = 2 * (ny * gx + nx) + d;
                        a[i, i] += GridSmooth; a[j, j] += GridSmooth;
                        a[i, j] -= GridSmooth; a[j, i] -= GridSmooth;
                    }
                }
            }
            var o = CalibrationSolver.SolveLinear(a, b);
            if (o == null) { log.Append("grid: no solution; "); return; }
            var grid = new float[u];
            float max = 0;
            int filled = 0;
            for (int i = 0; i < u; i++)
            {
                grid[i] = Mathf.Clamp((float)o[i], -GridMax, GridMax);
                max = Mathf.Max(max, Mathf.Abs(grid[i]));
            }
            foreach (int n in perCell) if (n >= GridMinPoints) filled++;
            lens.Grid = grid;
            log.Append($"grid: {filled}/{cells} cells with data, largest offset {max:F2} px; ");
        }

        // Zone of a thermal pixel by its distance from the image centre (0 centre .. 1 edge).
        static int ZoneOf(Vector2 px)
        {
            float d = Mathf.Max(Mathf.Abs(px.x / ThermalCamera.Width - 0.5f), Mathf.Abs(px.y / ThermalCamera.Height - 0.5f)) * 2;
            return d < 0.5f ? 0 : d < 0.8f ? 1 : 2;
        }

        static Func<Vector3, Vector2?> Projector(ThermalLens lens) => c => lens.RgbToPixel(c, out var px) ? px : null;

        // Turns the rotation by up to rangeDeg about the x and y axes (step stepDeg) to where the
        // most RGB edge points fall within 2 px of a thermal edge; returns how far it turned.
        static float Search(Frame[] frames, double[] p, float rangeDeg, float stepDeg, out int found)
        {
            var baseRotation = CalibrationSolver.ToQuaternion(p[0], p[1], p[2]);
            var sample = new List<(Vector3 c, ThermalEdges e)>();
            foreach (var f in frames)
                for (int i = 0; i < f.Points.Length; i += Math.Max(1, f.Points.Length / 300)) sample.Add((f.Points[i], f.Edges));
            int best = -1;
            Quaternion bestRotation = baseRotation;
            var q = (double[])p.Clone();
            for (float pitch = -rangeDeg; pitch <= rangeDeg + 1e-3f; pitch += stepDeg)
            for (float yaw = -rangeDeg; yaw <= rangeDeg + 1e-3f; yaw += stepDeg)
            {
                var r = Quaternion.Euler(pitch, yaw, 0) * baseRotation;
                SetRotation(q, r);
                CalibrationSolver.Rodrigues(q[0], q[1], q[2], out var m);
                int n = 0;
                foreach (var (c, e) in sample)
                {
                    if (!Project(q, m, c, out double x, out double y)) continue;
                    var px = new Vector2((float)x, (float)y);
                    if (e.Nearest(px, out var edge, out var normal) && Mathf.Abs(Vector2.Dot(normal, px - edge)) < 2 && (px - edge).sqrMagnitude < 9) n++;
                }
                if (n > best) { best = n; bestRotation = r; }
            }
            SetRotation(p, bestRotation);
            found = best;
            return Quaternion.Angle(bestRotation, baseRotation);
        }

        static void SetRotation(double[] p, Quaternion r)
        {
            r.ToAngleAxis(out float angle, out var axis);
            if (angle > 180) angle -= 360;
            var w = axis * (angle * Mathf.Deg2Rad);
            p[0] = w.x; p[1] = w.y; p[2] = w.z;
        }

        // How far, across the edge, the RGB edge points land from the thermal edges, per zone.
        static Zone[] Scores(Frame[] frames, Func<Vector3, Vector2?> project)
        {
            var errors = new[] { new List<float>(), new List<float>(), new List<float>() };
            foreach (var f in frames)
            foreach (var c in f.Points)
            {
                if (project(c) is not { } px) continue;
                if (!f.Edges.Nearest(px, out var edge, out var normal) || (px - edge).sqrMagnitude > ScoreGate * ScoreGate) continue;
                errors[ZoneOf(px)].Add(Mathf.Abs(Vector2.Dot(normal, px - edge)));
            }
            var zones = new Zone[3];
            for (int z = 0; z < 3; z++)
            {
                var e = errors[z];
                if (e.Count == 0) continue;
                e.Sort();
                int within = 0;
                foreach (float v in e) if (v < 1) within++;
                zones[z] = new Zone { Count = e.Count, MedianPx = e[e.Count / 2], Within1 = (float)within / e.Count };
            }
            return zones;
        }

        void OnDestroy()
        {
            if (_panel != null && _panel.Thermal != null) _panel.Thermal.FrameReady -= OnThermal;
            _points?.Release();
            _count?.Release();
            _points = _count = null;
        }
    }

    /// <summary>
    /// Edges of one thermal frame (°C, row 0 at the bottom): gradient ridges steeper than a
    /// threshold, at sub-pixel positions across the edge, and for every pixel the nearest one
    /// (two-pass propagation), for matching points to them.
    /// </summary>
    public class ThermalEdges
    {
        const int W = ThermalCamera.Width, H = ThermalCamera.Height;
        readonly Vector2[] _position;           // per edge pixel index
        readonly Vector2[] _normal;
        readonly int[] _nearest;                // per pixel: edge pixel index, -1 none

        public int Count { get; }

        public ThermalEdges(float[] celsius, float threshold)
        {
            var gx = new float[W * H];
            var gy = new float[W * H];
            var mag = new float[W * H];
            for (int y = 1; y < H - 1; y++)
            for (int x = 1; x < W - 1; x++)
            {
                int i = y * W + x;
                gx[i] = 0.5f * (celsius[i + 1] - celsius[i - 1]);
                gy[i] = 0.5f * (celsius[i + W] - celsius[i - W]);
                mag[i] = Mathf.Sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
            }
            _position = new Vector2[W * H];
            _normal = new Vector2[W * H];
            _nearest = new int[W * H];
            for (int i = 0; i < _nearest.Length; i++) _nearest[i] = -1;
            int count = 0;
            for (int y = 2; y < H - 2; y++)
            for (int x = 2; x < W - 2; x++)
            {
                int i = y * W + x;
                float m = mag[i];
                if (m < threshold) continue;
                var n = new Vector2(gx[i], gy[i]) / m;
                float before = Sample(mag, x - n.x, y - n.y), after = Sample(mag, x + n.x, y + n.y);
                if (m < before || m < after) continue;
                // Peak of a parabola through the three magnitudes, along the normal.
                float den = before - 2 * m + after;
                float offset = den < -1e-6f ? Mathf.Clamp(0.5f * (before - after) / den, -0.5f, 0.5f) : 0;
                _position[i] = new Vector2(x + 0.5f, y + 0.5f) + n * offset;
                _normal[i] = n;
                _nearest[i] = i;
                count++;
            }
            Count = count;
            Propagate();
        }

        static float Sample(float[] a, float x, float y)
        {
            int ix = Mathf.FloorToInt(x), iy = Mathf.FloorToInt(y);
            float fx = x - ix, fy = y - iy;
            float b = Mathf.Lerp(a[iy * W + ix], a[iy * W + ix + 1], fx);
            float t = Mathf.Lerp(a[(iy + 1) * W + ix], a[(iy + 1) * W + ix + 1], fx);
            return Mathf.Lerp(b, t, fy);
        }

        // Forward and backward raster passes over the 8 neighbours, keeping the nearest edge.
        void Propagate()
        {
            for (int pass = 0; pass < 2; pass++)
            {
                int start = pass == 0 ? 0 : H - 1, end = pass == 0 ? H : -1, step = pass == 0 ? 1 : -1;
                for (int y = start; y != end; y += step)
                for (int x = pass == 0 ? 0 : W - 1; x != (pass == 0 ? W : -1); x += step)
                {
                    int i = y * W + x;
                    var centre = new Vector2(x + 0.5f, y + 0.5f);
                    float best = _nearest[i] >= 0 ? (_position[_nearest[i]] - centre).sqrMagnitude : float.MaxValue;
                    for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int nx = x - dx * step, ny = y - dy * step;
                        if ((dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                        int k = _nearest[ny * W + nx];
                        if (k < 0) continue;
                        float d = (_position[k] - centre).sqrMagnitude;
                        if (d < best) { best = d; _nearest[i] = k; }
                    }
                }
            }
        }

        /// <summary>Nearest edge to a thermal pixel position (sub-pixel) and its normal; false off the image or without edges.</summary>
        public bool Nearest(Vector2 px, out Vector2 edge, out Vector2 normal)
        {
            edge = normal = default;
            int x = Mathf.FloorToInt(px.x), y = Mathf.FloorToInt(px.y);
            if (x < 0 || y < 0 || x >= W || y >= H) return false;
            int k = _nearest[y * W + x];
            if (k < 0) return false;
            edge = _position[k];
            normal = _normal[k];
            return true;
        }
    }
}
