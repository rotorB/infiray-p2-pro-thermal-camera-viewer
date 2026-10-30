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

        /// <summary>Which calibration projects the thermal image (when it exists; else the next one down).</summary>
        public enum Kind { Mapping, Lens, Edge, EdgeGrid, Nominal }
        public static readonly string[] KindNames = { "Mapping", "Lens (points)", "Edge", "Edge + grid", "Nominal (mount only)" };
        const int Calibrated = 4;       // kinds made by calibrating, below Nominal

        public int Version = 1;
        public List<Point> Points = new();
        /// <summary>Lens fitted to the Points wizard's pairs.</summary>
        public ThermalLens Lens = new();
        /// <summary>Lens and correction grid fitted by Edge align.</summary>
        public ThermalLens EdgeLens = new();
        /// <summary>Chosen calibration (<see cref="Kind"/>).</summary>
        public int Use = (int)Kind.EdgeGrid;
        /// <summary>
        /// How the thermal camera is mounted, turned about its view axis (0, 90, 180, 270): the
        /// root of every projection. Calibrations remember the mount they were made with and are
        /// turned by the difference, so changing it turns the thermal image at once; with no
        /// calibration the nominal lens is turned by it.
        /// </summary>
        public int MountRollDeg;
        /// <summary>Mount the mapping points were made with (their rolls include it).</summary>
        public int MappingMountDeg;

        static readonly ThermalLens Off = new();
        ThermalLens _remounted, _remountedFrom, _nominal;
        Quaternion _remountedRotation;

        /// <summary>Calibrated by the mapping points or a lens model.</summary>
        public bool Ready => Points.Count > 0 || Lens.Solved || EdgeLens.Solved;

        public bool Has(Kind kind) => kind switch
        {
            Kind.Mapping => Points.Count > 0,
            Kind.Lens => Lens.Solved,
            Kind.Edge => EdgeLens.Solved,
            Kind.EdgeGrid => EdgeLens.Solved && EdgeLens.HasGrid,
            _ => true,
        };

        /// <summary>
        /// The calibration in use: the chosen one if it exists, else the best one below it, else
        /// above; Nominal when chosen or when nothing is calibrated.
        /// </summary>
        public Kind Effective
        {
            get
            {
                if (Use == (int)Kind.Nominal) return Kind.Nominal;
                int use = Mathf.Clamp(Use, 0, Calibrated - 1);
                // An edge choice never falls back to the mapping or the points lens: the nominal lens instead.
                int lowest = use >= (int)Kind.Edge ? (int)Kind.Edge : 0;
                for (int k = use; k >= lowest; k--) if (Has((Kind)k)) return (Kind)k;
                if (lowest > 0) return Kind.Nominal;
                for (int k = use + 1; k < Calibrated; k++) if (Has((Kind)k)) return (Kind)k;
                return Kind.Nominal;
            }
        }

        public string EffectiveName => KindNames[(int)Effective];

        /// <summary>
        /// The lens that projects the thermal image now: the Points one, the Edge one (with or
        /// without its grid), or an unsolved one when the mapping is used.
        /// </summary>
        public ThermalLens Active
        {
            get
            {
                switch (Effective)
                {
                    case Kind.Lens: return Remount(Lens);
                    case Kind.Edge: EdgeLens.UseGrid = false; return Remount(EdgeLens);
                    case Kind.EdgeGrid: EdgeLens.UseGrid = true; return Remount(EdgeLens);
                    case Kind.Nominal:
                        if (_nominal == null || _nominal.MountDeg != MountRollDeg) _nominal = ThermalLens.Nominal(MountRollDeg);
                        return _nominal;
                    default: return Off;
                }
            }
        }

        // The lens as made, or turned to the current mount (a copy, kept while nothing changes).
        ThermalLens Remount(ThermalLens lens)
        {
            if (lens.MountDeg == MountRollDeg) return lens;
            if (_remounted == null || _remountedFrom != lens || _remountedRotation != lens.Rotation ||
                _remounted.MountDeg != MountRollDeg || _remounted.Fx != lens.Fx || _remounted.Grid.Length != lens.Grid.Length)
            {
                _remounted = lens.Remounted(MountRollDeg);
                _remountedFrom = lens;
                _remountedRotation = lens.Rotation;
            }
            _remounted.UseGrid = lens.UseGrid;
            return _remounted;
        }

        /// <summary>Next calibration that exists, from the current choice (Nominal included).</summary>
        public void NextKind()
        {
            int n = KindNames.Length;
            for (int i = 1; i <= n; i++)
            {
                var k = (Kind)((Use + i) % n);
                if (Has(k)) { Use = (int)k; return; }
            }
        }

        /// <summary>Deletes the mapping points, the lens and the edge calibration; the mount stays.</summary>
        public void ResetCalibrations()
        {
            Points.Clear();
            Lens = new ThermalLens();
            EdgeLens = new ThermalLens();
            Use = (int)Kind.EdgeGrid;
            _remounted = null;
        }

        /// <summary>
        /// Thermal texture coordinate of a point in the RGB camera's frame, by the lens model or
        /// else the mapping at the point's distance. False when not calibrated or not in view.
        /// </summary>
        public bool RgbToThermalUv(Vector3 rgb, Meta.XR.PassthroughCameraAccess source, out Vector2 uv)
        {
            uv = default;
            var lens = Active;
            if (lens.Solved)
            {
                if (!lens.RgbToPixel(rgb, out var px)) return false;
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
            var lens = Active;
            if (lens.Solved) ray = lens.PixelToRgbRay(new Vector2(uv.x * ThermalCamera.Width, uv.y * ThermalCamera.Height));
            else if (TryEvaluate(depth, out var m))
                ray = source.ViewportPointToRay(m.ThermalToRgb(uv), new Pose(Vector3.zero, Quaternion.identity));
            else return false;
            if (ray.direction.z < 0.05f) return false;
            rgb = ray.origin + ray.direction * ((depth - ray.origin.z) / ray.direction.z);
            return true;
        }

        /// <summary>
        /// The calibration in use as the debug panel's overlay at a distance: the mapping itself,
        /// or for a lens model the mapping that puts the thermal image's centre and a line across
        /// it where the lens does (its distortion and grid are not shown). False when not calibrated.
        /// </summary>
        public bool TryOverlay(float distanceM, Meta.XR.PassthroughCameraAccess source, out Mapping m)
        {
            m = default;
            if (!Active.Solved) return TryEvaluate(distanceM, out m);
            var identity = new Pose(Vector3.zero, Quaternion.identity);
            if (!View(new Vector2(0.5f, 0.5f), out var c) || !View(new Vector2(0.3f, 0.5f), out var l) ||
                !View(new Vector2(0.7f, 0.5f), out var r)) return false;
            var d = new Vector2(r.x - l.x, (r.y - l.y) * RgbAspect);    // RGB widths, as in Mapping
            m = new Mapping { U = c.x, V = c.y, Scale = d.magnitude / 0.4f, RollDeg = Mathf.Atan2(d.y, d.x) * Mathf.Rad2Deg };
            return true;

            bool View(Vector2 uv, out Vector2 viewport)
            {
                viewport = default;
                if (!ThermalUvToRgb(uv, distanceM, source, out var p)) return false;
                viewport = source.WorldToViewportPoint(p, identity);
                return true;
            }
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
            a.w += roll0 + MountRollDeg - MappingMountDeg;     // turned to the current mount
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
            if (Points.Count == 0) MappingMountDeg = MountRollDeg;
            mapping.RollDeg -= MountRollDeg - MappingMountDeg;  // stored at the mount the points were made with
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
