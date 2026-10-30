using System;
using System.Collections.Generic;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Four corners of a target in image coordinates: x right, y up from the bottom edge, pixel i
    /// covering [i, i+1). Order: top-left, top-right, bottom-right, bottom-left.
    /// </summary>
    public struct Quad
    {
        public Vector2 C0, C1, C2, C3;

        public Vector2 this[int i]
        {
            get => i switch { 0 => C0, 1 => C1, 2 => C2, _ => C3 };
            set { switch (i) { case 0: C0 = value; break; case 1: C1 = value; break; case 2: C2 = value; break; default: C3 = value; break; } }
        }

        public float Area => Mathf.Abs(Shoelace(C0, C1, C2, C3));
        public Vector2 Center => (C0 + C1 + C2 + C3) / 4;

        public Quad Scaled(float s) => new() { C0 = C0 * s, C1 = C1 * s, C2 = C2 * s, C3 = C3 * s };

        internal static float Shoelace(Vector2 a, Vector2 b, Vector2 c, Vector2 d) =>
            0.5f * (a.x * b.y - b.x * a.y + b.x * c.y - c.x * b.y + c.x * d.y - d.x * c.y + d.x * a.y - a.x * d.y);
    }

    /// <summary>
    /// Finds a bright (or warm) quadrilateral target: Otsu threshold, largest blob not touching the
    /// border, max-area quad on its convex hull, then sub-pixel corners from lines fitted to the
    /// threshold crossings along each side. Works on a single-channel float image, row 0 = bottom
    /// (°C for the thermal camera, luma for RGB).
    /// </summary>
    public class QuadDetector
    {
        public int MinArea = 100;          // blob pixels
        public float MinFill = 0.85f;      // blob area / quad area
        public int Border = 2;             // blob must stay this far from the image edge
        public float MinSide = 8;          // pixels

        int[] _labels = Array.Empty<int>();
        int[] _stack = Array.Empty<int>();
        readonly List<Vector2> _boundary = new();
        readonly List<Vector2> _hull = new();
        readonly List<Vector2> _edge = new();

        /// <summary>Otsu threshold of the image; also returns its range.</summary>
        public static float Otsu(float[] img, int n, out float lo, out float hi)
        {
            lo = float.MaxValue; hi = float.MinValue;
            for (int i = 0; i < n; i++) { float v = img[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
            if (hi - lo < 1e-6f) return hi;
            const int bins = 256;
            Span<int> hist = stackalloc int[bins];
            float scale = (bins - 1) / (hi - lo);
            for (int i = 0; i < n; i++) hist[(int)((img[i] - lo) * scale)]++;
            double sumAll = 0;
            for (int b = 0; b < bins; b++) sumAll += (double)b * hist[b];
            double sumB = 0, wB = 0, best = -1;
            int bestBin = 0;
            for (int b = 0; b < bins; b++)
            {
                wB += hist[b];
                if (wB == 0) continue;
                double wF = n - wB;
                if (wF == 0) break;
                sumB += (double)b * hist[b];
                double mB = sumB / wB, mF = (sumAll - sumB) / wF;
                double between = wB * wF * (mB - mF) * (mB - mF);
                if (between > best) { best = between; bestBin = b; }
            }
            return lo + (bestBin + 0.5f) / scale;
        }

        /// <summary>
        /// Coarse detection on <paramref name="img"/>. <paramref name="threshold"/> separates the
        /// target (above) from the background.
        /// </summary>
        public bool DetectCoarse(float[] img, int w, int h, float threshold, out Quad quad, out string reason)
        {
            quad = default;
            int n = w * h;
            if (_labels.Length < n) { _labels = new int[n]; _stack = new int[n]; }
            Array.Clear(_labels, 0, n);

            int bestLabel = 0, bestArea = 0, label = 0;
            for (int start = 0; start < n; start++)
            {
                if (_labels[start] != 0 || img[start] <= threshold) continue;
                label++;
                int area = 0, sp = 0;
                bool touchesBorder = false;
                _stack[sp++] = start;
                _labels[start] = label;
                while (sp > 0)
                {
                    int i = _stack[--sp];
                    area++;
                    int x = i % w, y = i / w;
                    if (x < Border || y < Border || x >= w - Border || y >= h - Border) touchesBorder = true;
                    if (x > 0) Push(i - 1);
                    if (x < w - 1) Push(i + 1);
                    if (y > 0) Push(i - w);
                    if (y < h - 1) Push(i + w);
                }
                if (!touchesBorder && area > bestArea) { bestArea = area; bestLabel = label; }

                void Push(int j)
                {
                    if (_labels[j] != 0 || img[j] <= threshold) return;
                    _labels[j] = label;
                    _stack[sp++] = j;
                }
            }
            if (bestLabel == 0) { reason = "no target (or it touches the image edge)"; return false; }
            if (bestArea < MinArea) { reason = "target too small"; return false; }

            _boundary.Clear();
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x;
                if (_labels[i] != bestLabel) continue;
                if (_labels[i - 1] != bestLabel || _labels[i + 1] != bestLabel ||
                    _labels[i - w] != bestLabel || _labels[i + w] != bestLabel)
                    _boundary.Add(new Vector2(x + 0.5f, y + 0.5f));
            }
            ConvexHull(_boundary, _hull);
            if (_hull.Count < 4) { reason = "degenerate target"; return false; }

            quad = MaxAreaQuad(_hull);
            float quadArea = quad.Area;
            // Boundary pixel centres sit half a pixel inside the true edge.
            float fill = bestArea / Mathf.Max(quadArea + 0.5f * Perimeter(quad), 1);
            if (fill < MinFill) { reason = $"not a rectangle (fill {fill:F2})"; return false; }
            for (int k = 0; k < 4; k++)
                if (Vector2.Distance(quad[k], quad[(k + 1) % 4]) < MinSide) { reason = "side too short"; return false; }

            quad = Order(quad);
            reason = null;
            return true;
        }

        /// <summary>
        /// Moves each side of <paramref name="quad"/> onto the sub-pixel threshold crossings
        /// within <paramref name="radius"/> pixels and intersects the fitted lines.
        /// </summary>
        public bool Refine(float[] img, int w, int h, float threshold, float radius, ref Quad quad, out string reason)
        {
            var lines = new (Vector2 point, Vector2 dir)[4];
            for (int k = 0; k < 4; k++)
            {
                Vector2 p = quad[k], q = quad[(k + 1) % 4];
                Vector2 dir = (q - p).normalized;
                Vector2 normal = new(-dir.y, dir.x);
                float length = Vector2.Distance(p, q);
                int samples = Mathf.Clamp((int)(length / 2), 8, 60);
                _edge.Clear();
                for (int s = 0; s < samples; s++)
                {
                    float t = Mathf.Lerp(0.15f, 0.85f, (s + 0.5f) / samples);
                    Vector2 onLine = Vector2.Lerp(p, q, t);
                    if (Crossing(img, w, h, threshold, onLine, normal, radius, out float offset))
                        _edge.Add(onLine + normal * offset);
                }
                if (_edge.Count < 6 || !FitLine(_edge, out lines[k].point, out lines[k].dir))
                {
                    reason = "edge not found";
                    return false;
                }
            }
            var refined = new Quad();
            for (int k = 0; k < 4; k++)
            {
                var a = lines[(k + 3) % 4];
                var b = lines[k];
                if (!Intersect(a.point, a.dir, b.point, b.dir, out Vector2 corner) ||
                    Vector2.Distance(corner, quad[k]) > radius * 2)
                {
                    reason = "corner refinement failed";
                    return false;
                }
                refined[k] = corner;
            }
            quad = refined;
            reason = null;
            return true;
        }

        static bool Crossing(float[] img, int w, int h, float threshold, Vector2 origin, Vector2 normal, float radius,
            out float offset)
        {
            const float step = 0.25f;
            offset = 0;
            float bestAbs = float.MaxValue;
            float prevS = -radius;
            float prev = Sample(img, w, h, origin + normal * prevS) - threshold;
            bool found = false;
            for (float s = -radius + step; s <= radius + 1e-4f; s += step)
            {
                float cur = Sample(img, w, h, origin + normal * s) - threshold;
                if (prev * cur <= 0 && prev != cur)
                {
                    float at = prevS + step * prev / (prev - cur);
                    if (Mathf.Abs(at) < bestAbs) { bestAbs = Mathf.Abs(at); offset = at; found = true; }
                }
                prev = cur;
                prevS = s;
            }
            return found;
        }

        /// <summary>Bilinear sample at continuous coordinates (pixel centres at i + 0.5).</summary>
        static float Sample(float[] img, int w, int h, Vector2 p)
        {
            float x = Mathf.Clamp(p.x - 0.5f, 0, w - 1.001f), y = Mathf.Clamp(p.y - 0.5f, 0, h - 1.001f);
            int x0 = (int)x, y0 = (int)y;
            float fx = x - x0, fy = y - y0;
            int i = y0 * w + x0;
            float top = Mathf.Lerp(img[i + w], img[i + w + 1], fx);
            float bottom = Mathf.Lerp(img[i], img[i + 1], fx);
            return Mathf.Lerp(bottom, top, fy);
        }

        // Total least squares line through the points.
        static bool FitLine(List<Vector2> pts, out Vector2 mean, out Vector2 dir)
        {
            mean = Vector2.zero;
            foreach (var p in pts) mean += p;
            mean /= pts.Count;
            float sxx = 0, sxy = 0, syy = 0;
            foreach (var p in pts)
            {
                var d = p - mean;
                sxx += d.x * d.x; sxy += d.x * d.y; syy += d.y * d.y;
            }
            float angle = 0.5f * Mathf.Atan2(2 * sxy, sxx - syy);
            dir = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
            return sxx + syy > 1e-6f;
        }

        static bool Intersect(Vector2 p, Vector2 r, Vector2 q, Vector2 s, out Vector2 x)
        {
            float cross = r.x * s.y - r.y * s.x;
            x = default;
            if (Mathf.Abs(cross) < 1e-4f) return false;
            Vector2 qp = q - p;
            float t = (qp.x * s.y - qp.y * s.x) / cross;
            x = p + r * t;
            return true;
        }

        static void ConvexHull(List<Vector2> pts, List<Vector2> hull)
        {
            pts.Sort((a, b) => a.x != b.x ? a.x.CompareTo(b.x) : a.y.CompareTo(b.y));
            hull.Clear();
            for (int pass = 0; pass < 2; pass++)
            {
                int start = hull.Count;
                for (int idx = 0; idx < pts.Count; idx++)
                {
                    var p = pts[pass == 0 ? idx : pts.Count - 1 - idx];
                    while (hull.Count >= start + 2 && Cross(hull[^2], hull[^1], p) <= 0) hull.RemoveAt(hull.Count - 1);
                    hull.Add(p);
                }
                hull.RemoveAt(hull.Count - 1);
            }
        }

        static float Cross(Vector2 o, Vector2 a, Vector2 b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

        // Diameter pair, the farthest points on each side of it, then a few passes moving each
        // corner to the hull point that maximises the quad's area.
        static Quad MaxAreaQuad(List<Vector2> hull)
        {
            int n = hull.Count, ia = 0, ic = 0;
            float best = -1;
            for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
            {
                float d = (hull[i] - hull[j]).sqrMagnitude;
                if (d > best) { best = d; ia = i; ic = j; }
            }
            int ib = ia, id = ia;
            float maxS = float.MinValue, minS = float.MaxValue;
            for (int i = 0; i < n; i++)
            {
                float s = Cross(hull[ia], hull[ic], hull[i]);
                if (s > maxS) { maxS = s; ib = i; }
                if (s < minS) { minS = s; id = i; }
            }
            var idx = new[] { ia, ib, ic, id };
            for (int pass = 0; pass < 3; pass++)
            for (int k = 0; k < 4; k++)
            {
                float bestArea = Mathf.Abs(Area(hull, idx));
                for (int i = 0; i < n; i++)
                {
                    int keep = idx[k];
                    idx[k] = i;
                    float a = Mathf.Abs(Area(hull, idx));
                    if (a > bestArea) bestArea = a; else idx[k] = keep;
                }
            }
            return new Quad { C0 = hull[idx[0]], C1 = hull[idx[1]], C2 = hull[idx[2]], C3 = hull[idx[3]] };
        }

        static float Area(List<Vector2> hull, int[] idx) =>
            Quad.Shoelace(hull[idx[0]], hull[idx[1]], hull[idx[2]], hull[idx[3]]);

        static float Perimeter(Quad q)
        {
            float p = 0;
            for (int k = 0; k < 4; k++) p += Vector2.Distance(q[k], q[(k + 1) % 4]);
            return p;
        }

        /// <summary>Top-left first, then clockwise as seen in the image (y up).</summary>
        static Quad Order(Quad q)
        {
            var c = q.Center;
            var pts = new[] { q.C0, q.C1, q.C2, q.C3 };
            Array.Sort(pts, (a, b) => Mathf.Atan2(b.y - c.y, b.x - c.x).CompareTo(Mathf.Atan2(a.y - c.y, a.x - c.x)));
            int tl = 0;
            for (int i = 1; i < 4; i++)
                if (pts[i].x - pts[i].y < pts[tl].x - pts[tl].y) tl = i;
            return new Quad { C0 = pts[tl], C1 = pts[(tl + 1) % 4], C2 = pts[(tl + 2) % 4], C3 = pts[(tl + 3) % 4] };
        }
    }
}
