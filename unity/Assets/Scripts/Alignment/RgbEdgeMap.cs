using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace P2Pro
{
    /// <summary>
    /// The headset RGB camera's edges at the thermal camera's pixel size (P2ProRgbEdgeMap.compute),
    /// rebuilt for each new RGB image while <see cref="Enabled"/>: the image is averaged down so
    /// one pixel covers about what one thermal pixel does, then gradient ridges are kept. Text and
    /// pictures on a screen and other fine detail, which the thermal camera cannot see, mostly
    /// vanish; outlines of objects remain. Used by the Edges look and Edge align.
    /// </summary>
    public class RgbEdgeMap : MonoBehaviour
    {
        public const float MinThreshold = 0.01f, MaxThreshold = 0.2f;
        const string ThresholdKey = "p2pro.rgbEdgeThreshold";

        static readonly int RgbTexId = Shader.PropertyToID("_RgbTex");
        static readonly int LumaId = Shader.PropertyToID("_Luma");
        static readonly int LumaReadId = Shader.PropertyToID("_LumaRead");
        static readonly int EdgeOutId = Shader.PropertyToID("_EdgeOut");
        static readonly int SizeId = Shader.PropertyToID("_Size");
        static readonly int ThresholdId = Shader.PropertyToID("_Threshold");

        ThermalPanel _panel;
        ThermalAlignment _alignment;
        ComputeShader _shader;
        int _downsample, _edges;
        RenderTexture _luma, _edgeTex;
        int _lastFrame = -1;
        float _threshold;

        /// <summary>Rebuild on new RGB images.</summary>
        public bool Enabled { get; set; }
        /// <summary>Edge strength 0..1 over the RGB image's texture coordinates; null until built.</summary>
        public Texture Texture => _edgeTex;
        /// <summary>RGB pixels averaged into one edge-map pixel.</summary>
        public float Scale { get; private set; }

        /// <summary>Luminance step per edge-map pixel an edge needs (0..1 scale).</summary>
        public float Threshold
        {
            get => _threshold;
            set
            {
                _threshold = Mathf.Clamp(value, MinThreshold, MaxThreshold);
                PlayerPrefs.SetFloat(ThresholdKey, _threshold);
                _lastFrame = -1;
            }
        }

        public void Init(ThermalPanel panel, ThermalAlignment alignment)
        {
            _panel = panel;
            _alignment = alignment;
            _shader = Resources.Load<ComputeShader>("P2ProRgbEdgeMap");
            _downsample = _shader.FindKernel("Downsample");
            _edges = _shader.FindKernel("Edges");
            _threshold = Mathf.Clamp(PlayerPrefs.GetFloat(ThresholdKey, 0.05f), MinThreshold, MaxThreshold);
        }

        void Update()
        {
            var rgb = _panel.Rgb;
            var texture = rgb.Texture;
            if (!Enabled || texture == null || rgb.Source == null || rgb.FrameCount == _lastFrame) return;
            _lastFrame = rgb.FrameCount;

            // One thermal pixel in RGB texture pixels: the ratio of their focal lengths.
            var source = rgb.Source;
            var sensor = (Vector2)source.Intrinsics.SensorResolution;
            var crop = (Vector2)source.CurrentResolution / sensor;
            crop /= Mathf.Max(crop.x, crop.y);
            float rgbFx = source.Intrinsics.FocalLength.x * texture.width / (sensor.x * crop.x);
            var lens = _alignment.Active;
            float thermalFx = lens.Solved ? lens.Fx : ThermalCalibration.Nominal().Fx;
            Scale = Mathf.Clamp(rgbFx / thermalFx, 1, 8);
            int w = Mathf.Max(8, Mathf.RoundToInt(texture.width / Scale)), h = Mathf.Max(8, Mathf.RoundToInt(texture.height / Scale));
            if (_edgeTex == null || _edgeTex.width != w || _edgeTex.height != h)
            {
                if (_luma != null) _luma.Release();
                if (_edgeTex != null) _edgeTex.Release();
                _luma = Create("RGB luma (thermal scale)", w, h);
                _edgeTex = Create("RGB edges (thermal scale)", w, h);
                _edgeTex.filterMode = FilterMode.Bilinear;
            }

            _shader.SetInts(SizeId, w, h);
            _shader.SetFloat(ThresholdId, _threshold);
            _shader.SetTexture(_downsample, RgbTexId, texture);
            _shader.SetTexture(_downsample, LumaId, _luma);
            _shader.SetTexture(_edges, LumaReadId, _luma);
            _shader.SetTexture(_edges, EdgeOutId, _edgeTex);
            _shader.Dispatch(_downsample, (w + 7) / 8, (h + 7) / 8, 1);
            _shader.Dispatch(_edges, (w + 7) / 8, (h + 7) / 8, 1);
        }

        static RenderTexture Create(string name, int w, int h)
        {
            var rt = new RenderTexture(w, h, 0, GraphicsFormat.R16_SFloat)
            {
                name = name,
                enableRandomWrite = true,
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Point,
            };
            rt.Create();
            return rt;
        }

        void OnDestroy()
        {
            PlayerPrefs.Save();
            if (_luma != null) _luma.Release();
            if (_edgeTex != null) _edgeTex.Release();
        }
    }
}
