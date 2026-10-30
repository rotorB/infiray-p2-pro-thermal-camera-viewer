using UnityEngine;
using UnityEngine.Rendering;

namespace P2Pro
{
    /// <summary>
    /// How the thermal overlay on the room mesh is blended with the room (P2ProOverlay.hlsl):
    /// blend mode, opacity (slider), a hot-only / cold-only temperature filter with its threshold
    /// (slider) and surfaces or wireframe edges. Remembered between launches, except the
    /// wireframe: every launch starts on surfaces.
    /// </summary>
    public class OverlayStyle
    {
        public enum BlendKind { Alpha, Add, Screen, Multiply }
        public enum FilterKind { All, Hot, Cold }

        public const float MinOpacity = 0.1f;
        static readonly int AlphaId = Shader.PropertyToID("_Alpha");
        static readonly int ModeId = Shader.PropertyToID("_Mode");
        static readonly int FilterId = Shader.PropertyToID("_Filter");
        static readonly int ThresholdId = Shader.PropertyToID("_Threshold");
        static readonly int WireId = Shader.PropertyToID("_Wire");
        static readonly int SrcBlendId = Shader.PropertyToID("_SrcBlend");
        static readonly int DstBlendId = Shader.PropertyToID("_DstBlend");
        static readonly int SrcBlendAId = Shader.PropertyToID("_SrcBlendA");
        static readonly int DstBlendAId = Shader.PropertyToID("_DstBlendA");
        const string Key = "p2pro.overlay.";

        public BlendKind Blend { get; private set; }
        public float Opacity { get; private set; }
        public FilterKind Filter { get; private set; }
        public float Threshold { get; private set; }
        /// <summary>Draws only the mesh's triangle edges, each coloured by its temperature.</summary>
        public bool Wire { get; private set; }

        bool _thresholdSet;

        public OverlayStyle()
        {
            Blend = (BlendKind)PlayerPrefs.GetInt(Key + "blend", 0);
            Opacity = Mathf.Clamp(PlayerPrefs.GetFloat(Key + "opacityValue", 0.75f), MinOpacity, 1);
            Filter = (FilterKind)PlayerPrefs.GetInt(Key + "filter", 0);
            Threshold = PlayerPrefs.GetFloat(Key + "threshold", 30);
            _thresholdSet = PlayerPrefs.HasKey(Key + "threshold");
        }

        public void NextBlend() { Blend = (BlendKind)(((int)Blend + 1) % 4); Save(); }
        public void SetOpacity(float opacity) { Opacity = Mathf.Clamp(opacity, MinOpacity, 1); Save(); }
        public void ToggleWire() => Wire = !Wire;

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
        public string WireLabel => Wire ? "Mesh: wire" : "Mesh: surface";
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
            PlayerPrefs.Save();
        }
    }
}
