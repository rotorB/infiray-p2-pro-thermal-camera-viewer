using UnityEngine;
using UnityEngine.Rendering;

namespace P2Pro
{
    /// <summary>
    /// How the thermal overlay on the room mesh is blended with the room (P2ProOverlay.hlsl):
    /// blend mode, opacity (slider), a hot-only / cold-only temperature filter with its threshold
    /// (slider) and the look: surfaces, wireframe edges, or the thermal image's edges as lines
    /// (to check the alignment against real outlines; Live and Flat). Remembered between
    /// launches, except the look: every launch starts on surfaces.
    /// </summary>
    public class OverlayStyle
    {
        public enum BlendKind { Alpha, Add, Screen, Multiply }
        public enum FilterKind { All, Hot, Cold }
        public enum LookKind { Surface, Wire, Edges }
        public const float MinEdgeThreshold = 0.1f, MaxEdgeThreshold = 5;

        public const float MinOpacity = 0.1f;
        static readonly int AlphaId = Shader.PropertyToID("_Alpha");
        static readonly int ModeId = Shader.PropertyToID("_Mode");
        static readonly int FilterId = Shader.PropertyToID("_Filter");
        static readonly int ThresholdId = Shader.PropertyToID("_Threshold");
        static readonly int WireId = Shader.PropertyToID("_Wire");
        static readonly int EdgesId = Shader.PropertyToID("_Edges");
        static readonly int EdgeThresholdId = Shader.PropertyToID("_EdgeThreshold");
        static readonly int SrcBlendId = Shader.PropertyToID("_SrcBlend");
        static readonly int DstBlendId = Shader.PropertyToID("_DstBlend");
        static readonly int SrcBlendAId = Shader.PropertyToID("_SrcBlendA");
        static readonly int DstBlendAId = Shader.PropertyToID("_DstBlendA");
        const string Key = "p2pro.overlay.";

        public BlendKind Blend { get; private set; }
        public float Opacity { get; private set; }
        public FilterKind Filter { get; private set; }
        public float Threshold { get; private set; }
        public LookKind Look { get; private set; }
        /// <summary>Draws only the mesh's triangle edges, each coloured by its temperature.</summary>
        public bool Wire => Look == LookKind.Wire;
        /// <summary>Draws only the thermal image's edges, steeper than this many °C per camera pixel.</summary>
        public float EdgeThreshold { get; private set; }

        bool _thresholdSet;

        public OverlayStyle()
        {
            Blend = (BlendKind)PlayerPrefs.GetInt(Key + "blend", 0);
            Opacity = Mathf.Clamp(PlayerPrefs.GetFloat(Key + "opacityValue", 0.75f), MinOpacity, 1);
            Filter = (FilterKind)PlayerPrefs.GetInt(Key + "filter", 0);
            Threshold = PlayerPrefs.GetFloat(Key + "threshold", 30);
            _thresholdSet = PlayerPrefs.HasKey(Key + "threshold");
            EdgeThreshold = Mathf.Clamp(PlayerPrefs.GetFloat(Key + "edgeThreshold", 0.8f), MinEdgeThreshold, MaxEdgeThreshold);
        }

        public void NextBlend() { Blend = (BlendKind)(((int)Blend + 1) % 4); Save(); }
        public void SetOpacity(float opacity) { Opacity = Mathf.Clamp(opacity, MinOpacity, 1); Save(); }
        /// <summary>Surface -> wire -> edges.</summary>
        public void NextLook() => Look = (LookKind)(((int)Look + 1) % 3);
        public void SetLook(LookKind look) => Look = look;

        public void SetEdgeThreshold(float celsius)
        {
            EdgeThreshold = Mathf.Clamp(celsius, MinEdgeThreshold, MaxEdgeThreshold);
            Save();
        }

        /// <summary>All -> hot -> cold. The first time, the threshold starts at 3/4 of the range.</summary>
        public void NextFilter(float rangeMin, float rangeMax)
        {
            Filter = (FilterKind)(((int)Filter + 1) % 3);
            if (!_thresholdSet && Filter != FilterKind.All)
            {
                Threshold = Mathf.Round(rangeMin + 0.75f * (rangeMax - rangeMin));
                _thresholdSet = true;
            }
            Save();
        }

        /// <summary>Sets the threshold; with the filter off, it turns to hot-only so the change shows.</summary>
        public void SetThreshold(float celsius)
        {
            Threshold = celsius;
            _thresholdSet = true;
            if (Filter == FilterKind.All) Filter = FilterKind.Hot;
            Save();
        }

        public string ThresholdLabel => Filter switch
        {
            FilterKind.Cold => $"Threshold: below {Threshold:F1} C",
            FilterKind.Hot => $"Threshold: above {Threshold:F1} C",
            _ => $"Threshold {Threshold:F1} C (drag = hot only)",
        };

        public string BlendLabel => "Blend: " + Blend.ToString().ToLowerInvariant();
        public string LookLabel => "Look: " + Look.ToString().ToLowerInvariant();
        public string FilterLabel => Filter switch
        {
            FilterKind.Hot => $"Show: > {Threshold:F0} C",
            FilterKind.Cold => $"Show: < {Threshold:F0} C",
            _ => "Show: all",
        };

        public void Apply(Material material)
        {
            material.SetFloat(AlphaId, Opacity);
            material.SetFloat(ModeId, (float)Blend);
            material.SetFloat(FilterId, (float)Filter);
            material.SetFloat(ThresholdId, Threshold);
            material.SetFloat(WireId, Wire ? 1 : 0);
            material.SetFloat(EdgesId, Look == LookKind.Edges ? 1 : 0);
            material.SetFloat(EdgeThresholdId, EdgeThreshold);
            var (src, dst, srcA, dstA) = Blend switch
            {
                BlendKind.Alpha => (BlendMode.SrcAlpha, BlendMode.OneMinusSrcAlpha, BlendMode.One, BlendMode.OneMinusSrcAlpha),
                BlendKind.Add => (BlendMode.One, BlendMode.One, BlendMode.Zero, BlendMode.One),
                _ => (BlendMode.One, BlendMode.OneMinusSrcAlpha, BlendMode.One, BlendMode.OneMinusSrcAlpha),
            };
            material.SetFloat(SrcBlendId, (float)src);
            material.SetFloat(DstBlendId, (float)dst);
            material.SetFloat(SrcBlendAId, (float)srcA);
            material.SetFloat(DstBlendAId, (float)dstA);
        }

        void Save()
        {
            PlayerPrefs.SetInt(Key + "blend", (int)Blend);
            PlayerPrefs.SetFloat(Key + "opacityValue", Opacity);
            PlayerPrefs.SetInt(Key + "filter", (int)Filter);
            PlayerPrefs.SetFloat(Key + "threshold", Threshold);
            PlayerPrefs.SetFloat(Key + "edgeThreshold", EdgeThreshold);
            PlayerPrefs.Save();
        }
    }
}
