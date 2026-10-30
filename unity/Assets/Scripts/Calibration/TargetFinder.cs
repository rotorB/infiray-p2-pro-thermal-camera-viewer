using System;
using Unity.Collections;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Finds the calibration target, a rectangle warmer and brighter than its surroundings (a
    /// monitor or laptop screen showing a white page), in a thermal or an RGB image. Corners come
    /// in pixels, y up, ordered by <see cref="QuadDetector"/>. Not thread-safe (shared buffers).
    /// </summary>
    public class TargetFinder
    {
        const float MinThermalContrast = 2;     // °C
        const float MinRgbContrast = 30;        // luma levels
        const int RgbDownscale = 4;

        readonly QuadDetector _thermal = new() { MinArea = 100, Border = 2, MinSide = 8 };
        readonly QuadDetector _rgb = new() { MinArea = 150, Border = 2, MinSide = 10 };
        float[] _luma = Array.Empty<float>();
        float[] _lumaSmall = Array.Empty<float>();

        /// <summary>Target in a °C image (<see cref="ThermalCamera.Celsius"/> layout).</summary>
        public bool FindThermal(float[] celsius, out Quad quad, out string reason)
        {
            const int w = ThermalCamera.Width, h = ThermalCamera.Height;
            quad = default;
            float threshold = QuadDetector.Otsu(celsius, w * h, out float lo, out float hi);
            if (hi - lo < MinThermalContrast) { reason = "no temperature contrast"; return false; }
            return _thermal.DetectCoarse(celsius, w, h, threshold, out quad, out reason) &&
                   _thermal.Refine(celsius, w, h, threshold, 3, ref quad, out reason);
        }

        /// <summary>Target in an RGB image, row 0 at the bottom.</summary>
        public bool FindRgb(NativeArray<Color32> pixels, int w, int h, out Quad quad, out string reason)
        {
            int n = w * h;
            quad = default;
            if (_luma.Length != n) _luma = new float[n];
            for (int i = 0; i < n; i++)
            {
                var p = pixels[i];
                _luma[i] = 0.299f * p.r + 0.587f * p.g + 0.114f * p.b;
            }
            int sw = w / RgbDownscale, sh = h / RgbDownscale;
            if (_lumaSmall.Length != sw * sh) _lumaSmall = new float[sw * sh];
            for (int y = 0; y < sh; y++)
            for (int x = 0; x < sw; x++)
            {
                float s = 0;
                for (int dy = 0; dy < RgbDownscale; dy++)
                for (int dx = 0; dx < RgbDownscale; dx++)
                    s += _luma[(y * RgbDownscale + dy) * w + x * RgbDownscale + dx];
                _lumaSmall[y * sw + x] = s / (RgbDownscale * RgbDownscale);
            }

            float threshold = QuadDetector.Otsu(_lumaSmall, sw * sh, out float lo, out float hi);
            if (hi - lo < MinRgbContrast) { reason = "no brightness contrast"; return false; }
            if (!_rgb.DetectCoarse(_lumaSmall, sw, sh, threshold, out quad, out reason)) return false;
            quad = quad.Scaled(RgbDownscale);
            return _rgb.Refine(_luma, w, h, threshold, RgbDownscale * 1.5f, ref quad, out reason);
        }
    }
}
