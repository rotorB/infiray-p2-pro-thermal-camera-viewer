using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// The thermal camera as a real camera next to the headset RGB camera, fitted to point pairs
    /// the user marks in the Points wizard (<see cref="AlignMode"/>): its rotation and position
    /// in the RGB camera's frame, focal lengths, principal point and radial distortion (k1, k2).
    /// Unlike the Align-mode mapping (shift, scale and roll as A + B / distance) it follows the
    /// two lenses' different perspective and distortion out to the image edges.
    /// Frames: the RGB camera's as in <see cref="RgbProjection"/> (x right, y up, z forward);
    /// thermal pixels with y up, row 0 at the bottom of the temperature texture.
    /// On top of the model, an optional grid of offsets (<see cref="Grid"/>, fitted by
    /// <see cref="EdgeAlign"/>) corrects what it leaves, out to the image corners.
    /// Shaders: P2ProThermalLens.hlsl, uniforms set by <see cref="Apply(ComputeShader)"/>.
    /// </summary>
    [Serializable]
    public class ThermalLens
    {
        public const int MinPairs = 6;
        const float MaxRms = 4;                 // thermal pixels; worse fits are not used

        [Serializable]
        public struct Pair
        {
            public Vector3 Rgb;                 // point in the RGB camera's frame at capture, metres
            public Vector2 ThermalPx;           // where it is in the thermal image, pixels, y up
        }

        public List<Pair> Pairs = new();
        public bool Solved;
        public Vector3 Position;                // thermal camera in the RGB camera frame, metres
        public Quaternion Rotation = Quaternion.identity;
        public float Fx, Fy, Cx, Cy, K1, K2;
        public float RmsPx;
        public int Used;
        /// <summary>Mount roll (<see cref="ThermalAlignment.MountRollDeg"/>) the model was made with; its Rotation includes it.</summary>
        public int MountDeg;
        public const int GridX = 16, GridY = 12;
        /// <summary>Thermal-pixel offsets (x, y per cell, row 0 at the bottom) added to the model's pixel; empty = none.</summary>
        public float[] Grid = Array.Empty<float>();

        public bool HasGrid => Grid != null && Grid.Length == GridX * GridY * 2;
        /// <summary>Applies the grid (set by <see cref="ThermalAlignment.Active"/>; not saved).</summary>
        [NonSerialized] public bool UseGrid = true;
        bool GridOn => Solved && HasGrid && UseGrid;

        static readonly int OnId = Shader.PropertyToID("_LensOn");
        static readonly int Row0Id = Shader.PropertyToID("_LensRow0");
        static readonly int Row1Id = Shader.PropertyToID("_LensRow1");
        static readonly int Row2Id = Shader.PropertyToID("_LensRow2");
        static readonly int KId = Shader.PropertyToID("_LensK");
        static readonly int DId = Shader.PropertyToID("_LensD");
        static readonly int GridOnId = Shader.PropertyToID("_LensGridOn");
        static readonly int GridId = Shader.PropertyToID("_LensGrid");
        Vector4[] _gridUniform;

        /// <summary>Same model, pairs and grid as a separate copy.</summary>
        public ThermalLens Clone()
        {
            var c = (ThermalLens)MemberwiseClone();
            c.Pairs = new List<Pair>(Pairs);
            c.Grid = (float[])Grid?.Clone() ?? Array.Empty<float>();
            c._gridUniform = null;
            return c;
        }

        /// <summary>Takes the model and grid of <paramref name="other"/>; keeps this one's pairs.</summary>
        public void CopyModel(ThermalLens other)
        {
            Solved = other.Solved;
            Position = other.Position;
            Rotation = other.Rotation;
            MountDeg = other.MountDeg;
            Fx = other.Fx; Fy = other.Fy; Cx = other.Cx; Cy = other.Cy; K1 = other.K1; K2 = other.K2;
            RmsPx = other.RmsPx;
            Used = other.Used;
            Grid = (float[])other.Grid.Clone();
            _gridUniform = null;
        }

        /// <summary>Grid offset at a model pixel (bilinear between cell centres, as the shader).</summary>
        public Vector2 GridOffset(Vector2 px)
        {
            if (!HasGrid || !UseGrid) return Vector2.zero;
            float gx = px.x / ThermalCamera.Width * GridX - 0.5f, gy = px.y / ThermalCamera.Height * GridY - 0.5f;
            int ix = Mathf.FloorToInt(gx), iy = Mathf.FloorToInt(gy);
            float fx = gx - ix, fy = gy - iy;
            var bottom = Vector2.Lerp(Cell(ix, iy), Cell(ix + 1, iy), fx);
            var top = Vector2.Lerp(Cell(ix, iy + 1), Cell(ix + 1, iy + 1), fx);
            return Vector2.Lerp(bottom, top, fy);
        }

        Vector2 Cell(int x, int y)
        {
            int i = Mathf.Clamp(y, 0, GridY - 1) * GridX + Mathf.Clamp(x, 0, GridX - 1);
            return new Vector2(Grid[2 * i], Grid[2 * i + 1]);
        }

        /// <summary>Thermal pixel of a point in the RGB camera frame; false behind the camera or far outside the image.</summary>
        public bool RgbToPixel(Vector3 rgb, out Vector2 px)
        {
            var c = Quaternion.Inverse(Rotation) * (rgb - Position);
            px = default;
            if (c.z <= 0.05f) return false;
            float x = c.x / c.z, y = c.y / c.z, r2 = x * x + y * y;
            if (r2 > 1.5f) return false;
            float d = 1 + r2 * (K1 + r2 * K2);
            px = new Vector2(Cx + Fx * x * d, Cy + Fy * y * d);
            px += GridOffset(px);
            return true;
        }

        /// <summary>Ray of a thermal pixel in the RGB camera frame (origin at the thermal camera).</summary>
        public Ray PixelToRgbRay(Vector2 px)
        {
            // Undo the grid: the model pixel m with m + offset(m) = px.
            var model = px;
            for (int i = 0; i < 5 && GridOn; i++) model = px - GridOffset(model);
            px = model;
            float xd = (px.x - Cx) / Fx, yd = (px.y - Cy) / Fy, x = xd, y = yd;
            for (int i = 0; i < 10; i++)
            {
                float r2 = x * x + y * y, d = 1 + r2 * (K1 + r2 * K2);
                x = xd / d;
                y = yd / d;
            }
            return new Ray(Position, Rotation * new Vector3(x, y, 1).normalized);
        }

        public void Apply(ComputeShader shader)
        {
            Uniforms(out var on, out var r0, out var r1, out var r2, out var k, out var d);
            shader.SetFloat(OnId, on);
            shader.SetVector(Row0Id, r0);
            shader.SetVector(Row1Id, r1);
            shader.SetVector(Row2Id, r2);
            shader.SetVector(KId, k);
            shader.SetVector(DId, d);
            shader.SetFloat(GridOnId, GridOn ? 1 : 0);
            shader.SetVectorArray(GridId, GridUniform());
        }

        public void Apply(Material material)
        {
            Uniforms(out var on, out var r0, out var r1, out var r2, out var k, out var d);
            material.SetFloat(OnId, on);
            material.SetVector(Row0Id, r0);
            material.SetVector(Row1Id, r1);
            material.SetVector(Row2Id, r2);
            material.SetVector(KId, k);
            material.SetVector(DId, d);
            material.SetFloat(GridOnId, GridOn ? 1 : 0);
            material.SetVectorArray(GridId, GridUniform());
        }

        // Two cells per vector, as P2ProThermalLens.hlsl reads them; zeros without a grid.
        Vector4[] GridUniform()
        {
            if (_gridUniform != null) return _gridUniform;
            _gridUniform = new Vector4[GridX * GridY / 2];
            if (HasGrid)
                for (int i = 0; i < _gridUniform.Length; i++)
                    _gridUniform[i] = new Vector4(Grid[4 * i], Grid[4 * i + 1], Grid[4 * i + 2], Grid[4 * i + 3]);
            return _gridUniform;
        }

        // RGB camera frame -> thermal camera frame as rows with the offset in w: R^T (c - P).
        void Uniforms(out float on, out Vector4 r0, out Vector4 r1, out Vector4 r2, out Vector4 k, out Vector4 d)
        {
            on = Solved ? 1 : 0;
            var m = Matrix4x4.Rotate(Quaternion.Inverse(Rotation));
            var t = -(Quaternion.Inverse(Rotation) * Position);
            r0 = new Vector4(m.m00, m.m01, m.m02, t.x);
            r1 = new Vector4(m.m10, m.m11, m.m12, t.y);
            r2 = new Vector4(m.m20, m.m21, m.m22, t.z);
            k = new Vector4(Fx, Fy, Cx, Cy);
            d = new Vector4(K1, K2, ThermalCamera.Width, ThermalCamera.Height);
        }

        public override string ToString()
        {
            var e = Rotation.eulerAngles;
            return $"{Used}/{Pairs.Count} pairs, {RmsPx:F2} px RMS; f {Fx:F0} x {Fy:F0}, centre {Cx:F0}, {Cy:F0}, " +
                   $"k1 {K1:F3} k2 {K2:F3}, at {Position * 100:F1} cm, turned {Wrap(e.x):F1} / {Wrap(e.y):F1} / {Wrap(e.z):F1} deg";
        }

        static float Wrap(float deg) => deg > 180 ? deg - 360 : deg;

        // ---- Solving ----

        // Parameters: rotation vector (3), position (3), fx, fy, cx, cy, k1, k2.
        const int N = 12;

        /// <summary>
        /// Fits the camera to <see cref="Pairs"/>: rotation from four starting rolls (the camera
        /// may be mounted at any), then intrinsics, then position (only when the pairs span
        /// distances) and distortion; pairs far off the fit are dropped once. Sets
        /// <see cref="Solved"/> when the result is plausible; returns a log line.
        /// </summary>
        /// <summary>
        /// Nominal P2 Pro lens at the RGB camera, turned by the mount: the projection before any
        /// calibration, and Edge align's start from scratch.
        /// </summary>
        public static ThermalLens Nominal(int mountDeg)
        {
            var n = ThermalCalibration.Nominal();
            return new ThermalLens
            {
                Solved = true,
                Rotation = Quaternion.AngleAxis(mountDeg, Vector3.forward),
                Fx = n.Fx, Fy = n.Fy, Cx = n.Cx, Cy = n.Cy,
                MountDeg = mountDeg,
            };
        }

        /// <summary>Copy turned about the thermal camera's view axis to another mount roll.</summary>
        public ThermalLens Remounted(int mountDeg)
        {
            var c = Clone();
            c.Rotation = Rotation * Quaternion.AngleAxis(mountDeg - MountDeg, Vector3.forward);
            c.MountDeg = mountDeg;
            return c;
        }

        public string Solve()
        {
            var log = new StringBuilder();
            int n = Pairs.Count;
            if (n < MinPairs) return $"Need at least {MinPairs} point pairs ({n} so far).";

            var nominal = ThermalCalibration.Nominal();
            var use = new bool[n];
            for (int i = 0; i < n; i++) use[i] = true;

            double[] best = null;
            double bestRms = double.MaxValue;
            for (int roll = 0; roll < 360; roll += 90)
            {
                double a = roll * Math.PI / 180;
                var p = new double[] { 0, 0, a, 0, 0, 0, nominal.Fx, nominal.Fy, nominal.Cx, nominal.Cy, 0, 0 };
                double rms = Lm(p, Free(rotation: true), use);
                if (rms < bestRms) { bestRms = rms; best = p; }
            }
            log.Append($"rotation {bestRms:F1} px");
            var q = best;
            log.Append($", intrinsics {Lm(q, Free(rotation: true, intrinsics: true), use):F1} px");

            float near = float.MaxValue, far = 0;
            foreach (var pair in Pairs) { near = Mathf.Min(near, pair.Rgb.magnitude); far = Mathf.Max(far, pair.Rgb.magnitude); }
            bool position = n >= 8 && far > 1.4f * near;
            bool k1 = n >= 8, k2 = n >= 14;
            var full = Free(rotation: true, intrinsics: true, position: position, k1: k1, k2: k2);
            double final = Lm(q, full, use);
            log.Append($", full {final:F1} px");

            if (Trim(q, use))
            {
                final = Lm(q, full, use);
                log.Append($", trimmed {final:F1} px");
            }

            Used = 0;
            foreach (bool u in use) if (u) Used++;
            bool plausible = final <= MaxRms && Used >= MinPairs &&
                             q[6] > 0.6 * nominal.Fx && q[6] < 1.6 * nominal.Fx &&
                             q[7] > 0.6 * nominal.Fy && q[7] < 1.6 * nominal.Fy &&
                             Math.Abs(q[10]) < 1 && new Vector3((float)q[3], (float)q[4], (float)q[5]).magnitude < 0.2f;
            if (!plausible)
            {
                log.Append(" - not plausible, kept the previous model");
                return log.ToString();
            }

            Rotation = CalibrationSolver.ToQuaternion(q[0], q[1], q[2]);
            Position = new Vector3((float)q[3], (float)q[4], (float)q[5]);
            Fx = (float)q[6]; Fy = (float)q[7]; Cx = (float)q[8]; Cy = (float)q[9];
            K1 = (float)q[10]; K2 = (float)q[11];
            RmsPx = (float)final;
            Solved = true;
            Grid = Array.Empty<float>();        // fitted to the previous model
            _gridUniform = null;
            if (!position) log.Append(" (position not fitted: add pairs at other distances)");
            return log.ToString();
        }

        static bool[] Free(bool rotation = false, bool intrinsics = false, bool position = false, bool k1 = false, bool k2 = false) =>
            new[] { rotation, rotation, rotation, position, position, position, intrinsics, intrinsics, intrinsics, intrinsics, k1, k2 };

        void Residuals(double[] p, bool[] use, double[] r)
        {
            CalibrationSolver.Rodrigues(p[0], p[1], p[2], out var m);
            for (int i = 0; i < Pairs.Count; i++)
            {
                if (!use[i]) { r[2 * i] = r[2 * i + 1] = 0; continue; }
                var c = Pairs[i].Rgb;
                double dx = c.x - p[3], dy = c.y - p[4], dz = c.z - p[5];
                // Thermal camera coordinates: transpose(R) * (c - t).
                double tx = m[0] * dx + m[3] * dy + m[6] * dz;
                double ty = m[1] * dx + m[4] * dy + m[7] * dz;
                double tz = Math.Max(m[2] * dx + m[5] * dy + m[8] * dz, 1e-3);
                double x = tx / tz, y = ty / tz, r2 = x * x + y * y, d = 1 + r2 * (p[10] + r2 * p[11]);
                r[2 * i] = p[8] + p[6] * x * d - Pairs[i].ThermalPx.x;
                r[2 * i + 1] = p[9] + p[7] * y * d - Pairs[i].ThermalPx.y;
            }
        }

        // Drops pairs more than 3 x the median error (and over 3 px) off; true if any.
        bool Trim(double[] p, bool[] use)
        {
            var r = new double[Pairs.Count * 2];
            Residuals(p, use, r);
            var errors = new List<double>();
            for (int i = 0; i < Pairs.Count; i++) if (use[i]) errors.Add(Math.Sqrt(r[2 * i] * r[2 * i] + r[2 * i + 1] * r[2 * i + 1]));
            errors.Sort();
            double limit = Math.Max(3 * errors[errors.Count / 2], 3);
            bool any = false;
            for (int i = 0; i < Pairs.Count; i++)
            {
                if (!use[i] || Math.Sqrt(r[2 * i] * r[2 * i] + r[2 * i + 1] * r[2 * i + 1]) <= limit) continue;
                use[i] = false;
                any = true;
            }
            return any;
        }

        /// <summary>Levenberg-Marquardt over the free parameters; returns the RMS in pixels.</summary>
        double Lm(double[] p, bool[] free, bool[] use)
        {
            int m = Pairs.Count * 2, used = 0;
            foreach (bool u in use) if (u) used++;
            var idx = new List<int>();
            for (int j = 0; j < N; j++) if (free[j]) idx.Add(j);
            int f = idx.Count;
            var r = new double[m];
            var rTry = new double[m];
            var jac = new double[f][];
            for (int j = 0; j < f; j++) jac[j] = new double[m];
            var a = new double[f, f];
            var g = new double[f];
            var trial = new double[N];

            Residuals(p, use, r);
            double cost = Dot(r, r), lambda = 1e-3;
            for (int iter = 0; iter < 100; iter++)
            {
                for (int j = 0; j < f; j++)
                {
                    int pj = idx[j];
                    double h = 1e-6 * Math.Max(1, Math.Abs(p[pj]));
                    Array.Copy(p, trial, N);
                    trial[pj] += h;
                    Residuals(trial, use, rTry);
                    for (int i = 0; i < m; i++) jac[j][i] = (rTry[i] - r[i]) / h;
                }
                for (int j = 0; j < f; j++)
                {
                    g[j] = -Dot(jac[j], r);
                    for (int k = 0; k <= j; k++) a[j, k] = a[k, j] = Dot(jac[j], jac[k]);
                }

                bool improved = false;
                for (int attempt = 0; attempt < 10 && !improved; attempt++)
                {
                    var aa = new double[f, f];
                    for (int j = 0; j < f; j++)
                    for (int k = 0; k < f; k++)
                        aa[j, k] = a[j, k] + (j == k ? lambda * Math.Max(a[j, j], 1e-9) : 0);
                    var step = CalibrationSolver.SolveLinear(aa, (double[])g.Clone());
                    if (step == null) { lambda *= 10; continue; }
                    Array.Copy(p, trial, N);
                    for (int j = 0; j < f; j++) trial[idx[j]] += step[j];
                    Residuals(trial, use, rTry);
                    double newCost = Dot(rTry, rTry);
                    if (newCost < cost)
                    {
                        bool converged = cost - newCost < 1e-10 * cost;
                        Array.Copy(trial, p, N);
                        Array.Copy(rTry, r, m);
                        cost = newCost;
                        lambda = Math.Max(lambda / 3, 1e-9);
                        improved = true;
                        if (converged) return Math.Sqrt(cost / Math.Max(1, used));
                    }
                    else
                    {
                        lambda *= 4;
                    }
                }
                if (!improved) break;
            }
            return Math.Sqrt(cost / Math.Max(1, used));
        }

        static double Dot(double[] a, double[] b)
        {
            double s = 0;
            for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
            return s;
        }
    }
}
