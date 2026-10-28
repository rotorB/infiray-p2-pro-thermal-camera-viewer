using System;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// 256x1 palette textures, left = coldest. Same palettes as the Kotlin viewer
    /// (android/app/.../Palettes.kt).
    /// </summary>
    public static class Palettes
    {
        public static readonly string[] Names = { "Inferno", "Turbo", "Ironbow", "White hot", "Black hot" };

        static readonly Func<float, Vector3>[] Functions =
        {
            Inferno,
            Turbo,
            Ironbow,
            t => new Vector3(t, t, t),
            t => new Vector3(1 - t, 1 - t, 1 - t),
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

        static Vector3 Ironbow(float t)
        {
            for (int i = 1; i < IronbowStops.Length; i++)
            {
                var (t1, c1) = IronbowStops[i];
                if (t > t1) continue;
                var (t0, c0) = IronbowStops[i - 1];
                return Vector3.Lerp(c0, c1, (t - t0) / (t1 - t0)) / 255f;
            }
            return Vector3.one;
        }
    }
}
