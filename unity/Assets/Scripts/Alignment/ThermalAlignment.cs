using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Where the thermal image lands in the headset RGB image, as a function of the distance to
    /// what is being looked at. Built from control points the user confirms in Align mode at a few
    /// distances (near, mid, far).
    ///
    /// A mapping places the thermal image in the RGB viewport (x right, y up, 0..1) as a rotated,
    /// scaled rectangle: in units of the RGB image width,
    ///   rgb = (U, V * RgbAspect) + Scale * Rotate(RollDeg) * ((tu - 0.5), (tv - 0.5) * ThermalAspect)
    /// for a thermal texture coordinate (tu, tv). Between control points every parameter follows
    /// a + b / distance, the way parallax between two side-by-side cameras falls off.
    ///
    /// Once the Points wizard has solved <see cref="Lens"/>, that camera model is used instead
    /// everywhere a world point is projected into the thermal image; the mapping stays for the
    /// panel overlay in Align mode.
    /// </summary>
    [Serializable]
    public class ThermalAlignment
    {
        public const float RgbAspect = 960f / 1280f;
        public const float ThermalAspect = (float)ThermalCamera.Height / ThermalCamera.Width;

        [Serializable]
        public struct Mapping
        {
            public float U, V, Scale, RollDeg;

            /// <summary>RGB viewport point of a thermal texture coordinate.</summary>
            public Vector2 ThermalToRgb(Vector2 thermalUv)
            {
                var t = new Vector2(thermalUv.x - 0.5f, (thermalUv.y - 0.5f) * ThermalAspect);
                var w = new Vector2(U, V * RgbAspect) + Scale * Rotate(t, RollDeg);
                return new Vector2(w.x, w.y / RgbAspect);
            }

            /// <summary>Thermal texture coordinate of an RGB viewport point.</summary>
            public Vector2 RgbToThermal(Vector2 rgbViewport)
            {
                var w = new Vector2(rgbViewport.x - U, (rgbViewport.y - V) * RgbAspect);
                var t = Rotate(w, -RollDeg) / Scale;
                return new Vector2(t.x + 0.5f, t.y / ThermalAspect + 0.5f);
            }

            public override string ToString() => $"centre {U:F3}, {V:F3}  scale {Scale:F3}  roll {RollDeg:F1} deg";
        }

        [Serializable]
        public struct Point
        {
            public float DistanceM;
            public Mapping Mapping;
            public string Source;       // "manual" or "auto+manual"
        }

        public int Version = 1;
        public List<Point> Points = new();
        public ThermalLens Lens = new();

        /// <summary>Calibrated by either the mapping points or the lens model.</summary>
        public bool Ready => Points.Count > 0 || Lens.Solved;

        /// <summary>
        /// Thermal texture coordinate of a point in the RGB camera's frame, by the lens model or
        /// else the mapping at the point's distance. False when not calibrated or not in view.
        /// </summary>
        public bool RgbToThermalUv(Vector3 rgb, Meta.XR.PassthroughCameraAccess source, out Vector2 uv)
        {
            uv = default;
            if (Lens.Solved)
            {
                if (!Lens.RgbToPixel(rgb, out var px)) return false;
                uv = new Vector2(px.x / ThermalCamera.Width, px.y / ThermalCamera.Height);
                return true;
            }
            if (rgb.z <= 0.05f || !TryEvaluate(rgb.magnitude, out var m)) return false;
            uv = m.RgbToThermal(source.WorldToViewportPoint(rgb, new Pose(Vector3.zero, Quaternion.identity)));
            return true;
        }

        /// <summary>
        /// Point in the RGB camera's frame that a thermal texture coordinate sees at
        /// <paramref name="depth"/> metres ahead of the RGB camera. False when not calibrated.
        /// </summary>
        public bool ThermalUvToRgb(Vector2 uv, float depth, Meta.XR.PassthroughCameraAccess source, out Vector3 rgb)
        {
            rgb = default;
            Ray ray;
            if (Lens.Solved) ray = Lens.PixelToRgbRay(new Vector2(uv.x * ThermalCamera.Width, uv.y * ThermalCamera.Height));
            else if (TryEvaluate(depth, out var m))
                ray = source.ViewportPointToRay(m.ThermalToRgb(uv), new Pose(Vector3.zero, Quaternion.identity));
            else return false;
            if (ray.direction.z < 0.05f) return false;
            rgb = ray.origin + ray.direction * ((depth - ray.origin.z) / ray.direction.z);
            return true;
        }

        public static string FilePath => Path.Combine(Application.persistentDataPath, "thermal_alignment.json");

        /// <summary>The mapping at a distance; false without control points.</summary>
        public bool TryEvaluate(float distanceM, out Mapping m)
        {
            m = default;
            if (!TryGetCoefficients(out var a, out var b)) return false;
            var v = a + b / Mathf.Max(distanceM, 0.1f);
            m = new Mapping { U = v.x, V = v.y, Scale = v.z, RollDeg = v.w };
            return true;
        }

        /// <summary>
        /// Mapping parameters (U, V, Scale, RollDeg) as A + B / distance, fitted by least squares;
        /// B is zero while the points do not spread in distance. False without control points.
        /// </summary>
        public bool TryGetCoefficients(out Vector4 a, out Vector4 b)
        {
            a = b = default;
            if (Points.Count == 0) return false;
            float roll0 = Points[0].Mapping.RollDeg;
            Fit(p => p.Mapping.U, out a.x, out b.x);
            Fit(p => p.Mapping.V, out a.y, out b.y);
            Fit(p => p.Mapping.Scale, out a.z, out b.z);
            Fit(p => Mathf.DeltaAngle(roll0, p.Mapping.RollDeg), out a.w, out b.w);
            a.w += roll0;
            return true;
        }

        void Fit(Func<Point, float> value, out float a, out float b)
        {
            int n = Points.Count;
            double sx = 0, sy = 0;
            foreach (var p in Points) { sx += 1 / p.DistanceM; sy += value(p); }
            double mx = sx / n, my = sy / n, sxx = 0, sxy = 0;
            foreach (var p in Points)
            {
                double dx = 1 / p.DistanceM - mx;
                sxx += dx * dx;
                sxy += dx * (value(p) - my);
            }
            double slope = sxx > 1e-4 ? sxy / sxx : 0;
            a = (float)(my - slope * mx);
            b = (float)slope;
        }

        /// <summary>False for mappings that put the thermal image mostly outside the RGB one.</summary>
        public static bool Plausible(Mapping m) =>
            m.U > -0.25f && m.U < 1.25f && m.V > -0.25f && m.V < 1.25f && m.Scale > 0.1f && m.Scale < 2;

        public void Add(float distanceM, Mapping mapping, string source)
        {
            Points.Add(new Point { DistanceM = distanceM, Mapping = mapping, Source = source });
            Points.Sort((a, b) => a.DistanceM.CompareTo(b.DistanceM));
        }

        public void Save() => File.WriteAllText(FilePath, JsonUtility.ToJson(this, true));

        public static ThermalAlignment Load()
        {
            try
            {
                return File.Exists(FilePath) ? JsonUtility.FromJson<ThermalAlignment>(File.ReadAllText(FilePath)) : new ThermalAlignment();
            }
            catch (Exception e)
            {
                Debug.LogWarning("[P2Pro] could not read alignment: " + e.Message);
                return new ThermalAlignment();
            }
        }

        /// <summary>
        /// The mapping that best lays the thermal target corners onto the RGB ones (scale,
        /// rotation, shift), trying all four corner pairings since the camera may be mounted at
        /// any roll. Corners in pixels, y up. <paramref name="rmsPx"/> is in RGB pixels.
        /// </summary>
        public static Mapping FitCorners(Quad thermal, Quad rgb, Vector2Int rgbSize, out float rmsPx)
        {
            // Both sets in RGB-width units, thermal centred on its image centre.
            var t = new Vector2[4];
            var w = new Vector2[4];
            for (int k = 0; k < 4; k++)
            {
                t[k] = new Vector2(thermal[k].x / ThermalCamera.Width - 0.5f,
                                   (thermal[k].y / ThermalCamera.Height - 0.5f) * ThermalAspect);
                w[k] = new Vector2(rgb[k].x / rgbSize.x, rgb[k].y / rgbSize.y * RgbAspect);
            }

            Mapping best = default;
            rmsPx = float.MaxValue;
            for (int shift = 0; shift < 4; shift++)
            {
                // Complex least squares: w = a + b * t, b = scale * e^(i roll).
                Vector2 mt = default, mw = default;
                for (int k = 0; k < 4; k++) { mt += t[k] / 4; mw += w[(k + shift) % 4] / 4; }
                double br = 0, bi = 0, nn = 0;
                for (int k = 0; k < 4; k++)
                {
                    var p = t[k] - mt;
                    var q = w[(k + shift) % 4] - mw;
                    br += q.x * p.x + q.y * p.y;        // q * conj(p)
                    bi += q.y * p.x - q.x * p.y;
                    nn += p.x * p.x + p.y * p.y;
                }
                if (nn < 1e-12) continue;
                br /= nn; bi /= nn;
                var m = new Mapping
                {
                    Scale = (float)Math.Sqrt(br * br + bi * bi),
                    RollDeg = (float)(Math.Atan2(bi, br) * 180 / Math.PI),
                };
                var centre = mw - m.Scale * Rotate(mt, m.RollDeg);
                m.U = centre.x;
                m.V = centre.y / RgbAspect;

                double sum = 0;
                for (int k = 0; k < 4; k++)
                {
                    var d = (centre + m.Scale * Rotate(t[k], m.RollDeg) - w[(k + shift) % 4]) * rgbSize.x;
                    sum += d.sqrMagnitude;
                }
                float rms = (float)Math.Sqrt(sum / 4);
                if (rms < rmsPx) { rmsPx = rms; best = m; }
            }
            return best;
        }

        static Vector2 Rotate(Vector2 v, float deg)
        {
            float r = deg * Mathf.Deg2Rad, c = Mathf.Cos(r), s = Mathf.Sin(r);
            return new Vector2(c * v.x - s * v.y, s * v.x + c * v.y);
        }
    }
}
