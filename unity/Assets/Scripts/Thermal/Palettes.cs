using System;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// 256x1 palette textures, left = coldest. The first five are the Kotlin viewer's
    /// (android/app/.../Palettes.kt); the rest are approximations of common thermography
    /// palettes, plus Viridis (colour-blind safe) and Hot spots (grey, top fifth red to yellow).
    /// </summary>
    public static class Palettes
    {
        public static readonly string[] Names =
        {
            "Inferno", "Turbo", "Ironbow", "White hot", "Black hot",
            "Rainbow", "Arctic", "Lava", "Viridis", "Hot spots",
        };

        static readonly Func<float, Vector3>[] Functions =
        {
            Inferno,
            Turbo,
            t => Stops(IronbowStops, t),
            t => new Vector3(t, t, t),
            t => new Vector3(1 - t, 1 - t, 1 - t),
            t => Stops(RainbowStops, t),
            t => Stops(ArcticStops, t),
            t => Stops(LavaStops, t),
            Viridis,
            HotSpots,
        };

        public static Texture2D Build(int index)
        {
            var tex = new Texture2D(256, 1, TextureFormat.RGBA32, false)
            {
                name = "P2Pro palette " + Names[index],
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
            };
            var pixels = new Color32[256];
            for (int i = 0; i < 256; i++)
            {
                var c = Functions[index](i / 255f);
                pixels[i] = new Color(Mathf.Clamp01(c.x), Mathf.Clamp01(c.y), Mathf.Clamp01(c.z));
            }
            tex.SetPixels32(pixels);
            tex.Apply(false, true);
            return tex;
        }

        // Polynomial fit of matplotlib's inferno (Matt Zucker, CC0).
        static Vector3 Inferno(float t)
        {
            Vector3[] c =
            {
                new(0.00021894037f, 0.0016510046f, -0.019480898f),
                new(0.10651342f, 0.56395644f, 3.9327124f),
                new(11.602493f, -3.972854f, -15.942394f),
                new(-41.703995f, 17.436399f, 44.354145f),
                new(77.16294f, -33.40236f, -81.80731f),
                new(-71.31943f, 32.626064f, 73.20952f),
                new(25.131126f, -12.242669f, -23.070325f),
            };
            var v = c[6];
            for (int k = 5; k >= 0; k--) v = v * t + c[k];
            return v;
        }

        // Polynomial approximation of Google's Turbo (Apache 2.0).
        static Vector3 Turbo(float t)
        {
            float r = 0.13572138f + t * (4.6153926f + t * (-42.66032f + t * (132.13108f + t * (-152.9424f + t * 59.28638f))));
            float g = 0.09140261f + t * (2.1941884f + t * (4.8429666f + t * (-14.185033f + t * (4.2772985f + t * 2.829566f))));
            float b = 0.1066733f + t * (12.641946f + t * (-60.582047f + t * (110.36277f + t * (-89.90311f + t * 27.34825f))));
            return new Vector3(r, g, b);
        }

        static readonly (float t, Vector3 rgb)[] IronbowStops =
        {
            (0.00f, new Vector3(0, 0, 0)),
            (0.15f, new Vector3(30, 0, 100)),
            (0.35f, new Vector3(130, 0, 150)),
            (0.55f, new Vector3(220, 60, 40)),
            (0.75f, new Vector3(255, 160, 0)),
            (0.90f, new Vector3(255, 230, 80)),
            (1.00f, new Vector3(255, 255, 255)),
        };

        static readonly (float t, Vector3 rgb)[] RainbowStops =
        {
            (0.00f, new Vector3(0, 0, 0)),
            (0.12f, new Vector3(40, 0, 120)),
            (0.25f, new Vector3(0, 60, 255)),
            (0.40f, new Vector3(0, 200, 220)),
            (0.55f, new Vector3(40, 220, 40)),
            (0.70f, new Vector3(255, 230, 0)),
            (0.85f, new Vector3(255, 60, 0)),
            (1.00f, new Vector3(255, 255, 255)),
        };

        // Cold surfaces in blues, warm ones in golds.
        static readonly (float t, Vector3 rgb)[] ArcticStops =
        {
            (0.00f, new Vector3(5, 5, 40)),
            (0.20f, new Vector3(20, 40, 130)),
            (0.40f, new Vector3(40, 110, 210)),
            (0.55f, new Vector3(120, 180, 235)),
            (0.70f, new Vector3(225, 160, 70)),
            (0.85f, new Vector3(255, 205, 60)),
            (1.00f, new Vector3(255, 250, 210)),
        };

        static readonly (float t, Vector3 rgb)[] LavaStops =
        {
            (0.00f, new Vector3(0, 0, 0)),
            (0.15f, new Vector3(20, 40, 90)),
            (0.30f, new Vector3(40, 100, 140)),
            (0.45f, new Vector3(120, 60, 160)),
            (0.60f, new Vector3(210, 40, 80)),
            (0.75f, new Vector3(250, 120, 20)),
            (0.90f, new Vector3(255, 210, 60)),
            (1.00f, new Vector3(255, 255, 230)),
        };

        static Vector3 Stops((float t, Vector3 rgb)[] stops, float t)
        {
            for (int i = 1; i < stops.Length; i++)
            {
                var (t1, c1) = stops[i];
                if (t > t1) continue;
                var (t0, c0) = stops[i - 1];
                return Vector3.Lerp(c0, c1, (t - t0) / (t1 - t0)) / 255f;
            }
            return stops[^1].rgb / 255f;
        }

        // Polynomial fit of matplotlib's viridis (Matt Zucker, CC0).
        static Vector3 Viridis(float t)
        {
            Vector3[] c =
            {
                new(0.2777273f, 0.0054073f, 0.3340998f),
                new(0.1050930f, 1.4046135f, 1.3845902f),
                new(-0.3308618f, 0.2148476f, 0.0950952f),
                new(-4.6342305f, -5.7991010f, -19.3324410f),
                new(6.2282699f, 14.1799334f, 56.6905526f),
                new(4.7763850f, -13.7451454f, -65.3530326f),
                new(-5.4354559f, 4.6458526f, 26.3124352f),
            };
            var v = c[6];
            for (int k = 5; k >= 0; k--) v = v * t + c[k];
            return v;
        }

        // Grey for most of the range, so the hottest fifth stands out in red to yellow.
        static Vector3 HotSpots(float t)
        {
            const float Start = 0.8f;
            if (t < Start)
            {
                float g = 0.1f + 0.7f * t / Start;
                return new Vector3(g, g, g);
            }
            float k = (t - Start) / (1 - Start);
            return new Vector3(1, k, k * k * 0.6f);
        }
    }
}
