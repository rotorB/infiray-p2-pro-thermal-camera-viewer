using UnityEngine;
using UnityEngine.Rendering;

namespace P2Pro
{
    /// <summary>
    /// How the thermal overlay on the room mesh is blended with the room (P2ProOverlay.hlsl):
    /// blend mode, opacity, a hot-only / cold-only temperature filter and surfaces or wireframe
    /// edges. Remembered between launches, except the wireframe: every launch starts on surfaces.
    /// </summary>
    public class OverlayStyle
    {
        public enum BlendKind { Alpha, Add, Screen, Multiply }
        public enum FilterKind { All, Hot, Cold }

        static readonly float[] Opacities = { 0.25f, 0.5f, 0.75f, 1 };
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
        public float Opacity => Opacities[_opacity];
        public FilterKind Filter { get; private set; }
        public float Threshold { get; private set; }
        /// <summary>Draws only the mesh's triangle edges, each coloured by its temperature.</summary>
        public bool Wire { get; private set; }

        int _opacity;
        bool _thresholdSet;

        public OverlayStyle()
        {
            Blend = (BlendKind)PlayerPrefs.GetInt(Key + "blend", 0);
            _opacity = Mathf.Clamp(PlayerPrefs.GetInt(Key + "opacity", 2), 0, Opacities.Length - 1);
            Filter = (FilterKind)PlayerPrefs.GetInt(Key + "filter", 0);
            Threshold = PlayerPrefs.GetFloat(Key + "threshold", 30);
            _thresholdSet = PlayerPrefs.HasKey(Key + "threshold");
        }

        public void NextBlend() { Blend = (BlendKind)(((int)Blend + 1) % 4); Save(); }
        public void NextOpacity() { _opacity = (_opacity + 1) % Opacities.Length; Save(); }
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

        public void NudgeThreshold(float celsius)
        {
            Threshold += celsius;
            _thresholdSet = true;
            Save();
        }

        public string BlendLabel => "Blend: " + Blend.ToString().ToLowerInvariant();
        public string OpacityLabel => $"Opacity {Opacity * 100:F0}%";
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
            PlayerPrefs.SetInt(Key + "opacity", _opacity);
            PlayerPrefs.SetInt(Key + "filter", (int)Filter);
            PlayerPrefs.SetFloat(Key + "threshold", Threshold);
            PlayerPrefs.Save();
        }
    }
}
