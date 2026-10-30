using System;
using System.Collections.Generic;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Fits <see cref="ThermalCalibration"/> to thermal target corners: rotation and position of
    /// the thermal camera in the head frame, fx, fy, cx, cy, k1, and the USB latency.
    /// Levenberg-Marquardt in double precision on reprojection error, in stages (rotation, then
    /// intrinsics, then everything), with outlier frames trimmed and the latency picked from a
    /// grid. Pure C#: safe to run on a worker thread.
    /// </summary>
    public static class CalibrationSolver
    {
        public class Observation
        {
            public readonly Vector2[] Corners = new Vector2[4];
            /// <summary>Head pose at arrival time minus i * <see cref="Input.LatencyStepMs"/>.</summary>
            public Pose[] HeadPoses;
        }

        public class Input
        {
            public Vector3[] Corners3D;             // world, same corner order as the observations
            public List<Observation> Observations;
            public float LatencyStepMs;
            public int RgbViews;
        }

        public class Result
        {
            public ThermalCalibration Calibration;
            public int Used;
            public int Total;
            public string Log;
        }

        const int MaxObservations = 150;
        const int NumParams = 11;               // rotation vector (3), position (3), fx, fy, cx, cy, k1

        static readonly bool[] RotationOnly = { true, true, true, false, false, false, false, false, false, false, false };
        static readonly bool[] RotationIntrinsics = { true, true, true, false, false, false, true, true, true, true, false };
        static readonly bool[] All = { true, true, true, true, true, true, true, true, true, true, true };

        public static Result Solve(Input input)
        {
            var log = new System.Text.StringBuilder();
            var obs = Subsample(input.Observations, MaxObservations);
            int n = obs.Count, latencies = obs[0].HeadPoses.Length;

            var observed = new double[n * 8];
            for (int i = 0; i < n; i++)
            for (int k = 0; k < 4; k++)
            {
                observed[i * 8 + k * 2] = obs[i].Corners[k].x;
                observed[i * 8 + k * 2 + 1] = obs[i].Corners[k].y;
            }

            var nominal = ThermalCalibration.Nominal();
            var p = new double[NumParams];
            p[6] = nominal.Fx; p[7] = nominal.Fy; p[8] = nominal.Cx; p[9] = nominal.Cy;

            // The thermal camera may be mounted at any roll, so its top-left corner can be any corner
            // of the target. Try the four corner assignments with four starting rolls each and keep
            // the best; negative focal lengths (a roll of 180 degrees in disguise, or a mirror) lose.
            int shift = 0;
            double[] start = null;
            double startRms = double.MaxValue;
            for (int s = 0; s < 4; s++)
            {
                var points = HeadPoints(Shift(input.Corners3D, s), obs, 0);
                for (int roll = 0; roll < 4; roll++)
                {
                    var q = (double[])p.Clone();
                    q[2] = (roll < 3 ? roll : -1) * Math.PI / 2;
                    var use = new bool[n];
                    Array.Fill(use, true);
                    Lm(q, RotationOnly, points, observed, use);
                    double rms = Lm(q, RotationIntrinsics, points, observed, use);
                    if (q[6] <= 0 || q[7] <= 0) continue;
                    if (rms < startRms) { startRms = rms; start = q; shift = s; }
                }
            }
            if (start == null) throw new InvalidOperationException("no corner assignment fits");
            var corners3D = Shift(input.Corners3D, shift);
            log.AppendLine($"corner shift {shift}, start RMS {startRms:F2} px");

            double[] bestP = null;
            bool[] bestUse = null;
            double bestRms = double.MaxValue;
            int bestLatency = 0;

            // Coarse latency grid (every other step), then the neighbours of the best.
            var tried = new HashSet<int>();
            var order = new List<int>();
            for (int l = 0; l < latencies; l += 2) order.Add(l);
            for (int pass = 0; pass < 2; pass++)
            {
                foreach (int l in order)
                {
                    if (!tried.Add(l)) continue;
                    var points = HeadPoints(corners3D, obs, l);
                    var use = new bool[n];
                    Array.Fill(use, true);
                    var q = (double[])(bestP ?? start).Clone();
                    Lm(q, All, points, observed, use);
                    Trim(q, points, observed, use);
                    double rms = Lm(q, All, points, observed, use);
                    log.AppendLine($"latency {l * input.LatencyStepMs:F0} ms: RMS {rms:F3} px, used {Count(use)}/{n}");
                    if (rms < bestRms) { bestRms = rms; bestP = q; bestUse = use; bestLatency = l; }
                }
                order.Clear();
                if (bestLatency > 0) order.Add(bestLatency - 1);
                if (bestLatency < latencies - 1) order.Add(bestLatency + 1);
            }

            var cal = new ThermalCalibration
            {
                Rotation = ToQuaternion(bestP[0], bestP[1], bestP[2]),
                Position = new Vector3((float)bestP[3], (float)bestP[4], (float)bestP[5]),
                Fx = (float)bestP[6], Fy = (float)bestP[7], Cx = (float)bestP[8], Cy = (float)bestP[9], K1 = (float)bestP[10],
                LatencyMs = bestLatency * input.LatencyStepMs,
                RmsPx = (float)bestRms,
                Samples = Count(bestUse),
                RgbViews = input.RgbViews,
            };
            return new Result { Calibration = cal, Used = Count(bestUse), Total = n, Log = log.ToString() };
        }

        // Corners3D[(k + shift) % 4]: which target corner the thermal image calls corner k.
        static Vector3[] Shift(Vector3[] corners, int shift)
        {
            var shifted = new Vector3[4];
            for (int k = 0; k < 4; k++) shifted[k] = corners[(k + shift) % 4];
            return shifted;
        }

        static List<Observation> Subsample(List<Observation> all, int max)
        {
            if (all.Count <= max) return all;
            var picked = new List<Observation>(max);
            for (int i = 0; i < max; i++) picked.Add(all[(int)((long)i * all.Count / max)]);
            return picked;
        }

        // Target corners in the head frame of every observation, for one latency.
        static double[] HeadPoints(Vector3[] world, List<Observation> obs, int latency)
        {
            var pts = new double[obs.Count * 12];
            for (int i = 0; i < obs.Count; i++)
            {
                var head = obs[i].HeadPoses[latency];
                var inv = Quaternion.Inverse(head.rotation);
                for (int k = 0; k < 4; k++)
                {
                    var h = inv * (world[k] - head.position);
                    pts[i * 12 + k * 3] = h.x; pts[i * 12 + k * 3 + 1] = h.y; pts[i * 12 + k * 3 + 2] = h.z;
                }
            }
            return pts;
        }

        static void Residuals(double[] p, double[] pts, double[] observed, bool[] use, double[] r)
        {
            Rodrigues(p[0], p[1], p[2], out var m);
            int n = use.Length;
            for (int i = 0; i < n; i++)
            {
                if (!use[i]) { for (int j = 0; j < 8; j++) r[i * 8 + j] = 0; continue; }
                for (int k = 0; k < 4; k++)
                {
                    int o = i * 12 + k * 3;
                    double dx = pts[o] - p[3], dy = pts[o + 1] - p[4], dz = pts[o + 2] - p[5];
                    // Camera coordinates: transpose(R) * (P - t).
                    double cx = m[0] * dx + m[3] * dy + m[6] * dz;
                    double cy = m[1] * dx + m[4] * dy + m[7] * dz;
                    double cz = m[2] * dx + m[5] * dy + m[8] * dz;
                    if (cz < 1e-3) cz = 1e-3;
                    double x = cx / cz, y = cy / cz, d = 1 + p[10] * (x * x + y * y);
                    r[i * 8 + k * 2] = p[8] + p[6] * x * d - observed[i * 8 + k * 2];
                    r[i * 8 + k * 2 + 1] = p[9] + p[7] * y * d - observed[i * 8 + k * 2 + 1];
                }
            }
        }

        /// <summary>Levenberg-Marquardt over the free parameters; returns the RMS in pixels.</summary>
        static double Lm(double[] p, bool[] free, double[] pts, double[] observed, bool[] use)
        {
            int m = observed.Length;
            var idx = new List<int>();
            for (int j = 0; j < NumParams; j++) if (free[j]) idx.Add(j);
            int f = idx.Count;
            var r = new double[m];
            var rTry = new double[m];
            var jac = new double[f][];
            for (int j = 0; j < f; j++) jac[j] = new double[m];
            var a = new double[f, f];
            var g = new double[f];
            var trial = new double[NumParams];

            Residuals(p, pts, observed, use, r);
            double cost = Dot(r, r), lambda = 1e-3;
            for (int iter = 0; iter < 60; iter++)
            {
                for (int j = 0; j < f; j++)
                {
                    int pj = idx[j];
                    double h = 1e-6 * Math.Max(1, Math.Abs(p[pj]));
                    Array.Copy(p, trial, NumParams);
                    trial[pj] += h;
                    Residuals(trial, pts, observed, use, rTry);
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
                    var step = SolveLinear(aa, (double[])g.Clone());
                    if (step == null) { lambda *= 10; continue; }
                    Array.Copy(p, trial, NumParams);
                    for (int j = 0; j < f; j++) trial[idx[j]] += step[j];
                    Residuals(trial, pts, observed, use, rTry);
                    double newCost = Dot(rTry, rTry);
                    if (newCost < cost)
                    {
                        bool converged = cost - newCost < 1e-10 * cost;
                        Array.Copy(trial, p, NumParams);
                        Array.Copy(rTry, r, m);
                        cost = newCost;
                        lambda = Math.Max(lambda / 3, 1e-9);
                        improved = true;
                        if (converged) return Rms(cost, use);
                    }
                    else
                    {
                        lambda *= 4;
                    }
                }
                if (!improved) break;
            }
            return Rms(cost, use);
        }

        // Drops frames whose RMS is far above the median (bad detections, target moved).
        static void Trim(double[] p, double[] pts, double[] observed, bool[] use)
        {
            var r = new double[observed.Length];
            Residuals(p, pts, observed, use, r);
            var perObs = new double[use.Length];
            var kept = new List<double>();
            for (int i = 0; i < use.Length; i++)
            {
                double s = 0;
                for (int j = 0; j < 8; j++) s += r[i * 8 + j] * r[i * 8 + j];
                perObs[i] = Math.Sqrt(s / 8);
                if (use[i]) kept.Add(perObs[i]);
            }
            kept.Sort();
            double limit = Math.Max(3 * kept[kept.Count / 2], 1.0);
            for (int i = 0; i < use.Length; i++)
                if (perObs[i] > limit) use[i] = false;
        }

        static double Rms(double cost, bool[] use) => Math.Sqrt(cost / Math.Max(1, Count(use) * 8));

        static int Count(bool[] use)
        {
            int c = 0;
            foreach (bool u in use) if (u) c++;
            return c;
        }

        static double Dot(double[] a, double[] b)
        {
            double s = 0;
            for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
            return s;
        }

        // Gaussian elimination with partial pivoting; null if singular.
        internal static double[] SolveLinear(double[,] a, double[] b)
        {
            int n = b.Length;
            for (int c = 0; c < n; c++)
            {
                int piv = c;
                for (int r = c + 1; r < n; r++) if (Math.Abs(a[r, c]) > Math.Abs(a[piv, c])) piv = r;
                if (Math.Abs(a[piv, c]) < 1e-15) return null;
                if (piv != c)
                {
                    for (int k = 0; k < n; k++) (a[c, k], a[piv, k]) = (a[piv, k], a[c, k]);
                    (b[c], b[piv]) = (b[piv], b[c]);
                }
                for (int r = c + 1; r < n; r++)
                {
                    double factor = a[r, c] / a[c, c];
                    for (int k = c; k < n; k++) a[r, k] -= factor * a[c, k];
                    b[r] -= factor * b[c];
                }
            }
            var x = new double[n];
            for (int r = n - 1; r >= 0; r--)
            {
                double s = b[r];
                for (int k = r + 1; k < n; k++) s -= a[r, k] * x[k];
                x[r] = s / a[r, r];
            }
            return x;
        }

        // Row-major rotation matrix of a rotation vector (axis * angle in radians).
        internal static void Rodrigues(double wx, double wy, double wz, out double[] m)
        {
            double theta = Math.Sqrt(wx * wx + wy * wy + wz * wz);
            m = new double[9];
            if (theta < 1e-12)
            {
                m[0] = m[4] = m[8] = 1;
                m[1] = -wz; m[2] = wy; m[3] = wz; m[5] = -wx; m[6] = -wy; m[7] = wx;
                return;
            }
            double kx = wx / theta, ky = wy / theta, kz = wz / theta;
            double c = Math.Cos(theta), s = Math.Sin(theta), v = 1 - c;
            m[0] = c + kx * kx * v;      m[1] = kx * ky * v - kz * s; m[2] = kx * kz * v + ky * s;
            m[3] = ky * kx * v + kz * s; m[4] = c + ky * ky * v;      m[5] = ky * kz * v - kx * s;
            m[6] = kz * kx * v - ky * s; m[7] = kz * ky * v + kx * s; m[8] = c + kz * kz * v;
        }

        // Same rotation as Rodrigues: Unity's quaternion product applies the standard formula.
        internal static Quaternion ToQuaternion(double wx, double wy, double wz)
        {
            double theta = Math.Sqrt(wx * wx + wy * wy + wz * wz);
            if (theta < 1e-12) return Quaternion.identity;
            return Quaternion.AngleAxis((float)(theta * 180 / Math.PI),
                new Vector3((float)(wx / theta), (float)(wy / theta), (float)(wz / theta)));
        }
    }
}
