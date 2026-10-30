using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace P2Pro
{
    /// <summary>
    /// Extra sharpness for the thermal image, ported from the macOS viewer (app.py, high quality
    /// mode): 3x bilinear upsample, then an unsharp mask with a Gaussian of sigma 1.2 upsampled
    /// pixels (P2ProSharpen.compute). Runs on °C on every new frame; <see cref="Output"/> is
    /// what the panel and the room overlays show. <see cref="ThermalCamera.Celsius"/> stays
    /// unsharpened for measurements. On/off is remembered between launches.
    /// </summary>
    public class ThermalSharpening : MonoBehaviour
    {
        public const int Upscale = 3;
        public const float Strength = 4;
        const float Sigma = 1.2f;
        const string Key = "p2pro.sharpen";

        static readonly int SourceId = Shader.PropertyToID("_Source");
        static readonly int UpId = Shader.PropertyToID("_Up");
        static readonly int TmpId = Shader.PropertyToID("_Tmp");
        static readonly int ResultId = Shader.PropertyToID("_Result");
        static readonly int UpReadId = Shader.PropertyToID("_UpRead");
        static readonly int TmpReadId = Shader.PropertyToID("_TmpRead");
        static readonly int SourceSizeId = Shader.PropertyToID("_SourceSize");
        static readonly int SizeId = Shader.PropertyToID("_Size");
        static readonly int SigmaId = Shader.PropertyToID("_Sigma");
        static readonly int RadiusId = Shader.PropertyToID("_Radius");
        static readonly int StrengthId = Shader.PropertyToID("_Strength");

        ThermalCamera _thermal;
        ComputeShader _shader;
        int _upsample, _blurX, _sharpen;
        RenderTexture _up, _tmp, _result;
        bool _enabled;

        /// <summary>Sharpened °C at <see cref="Upscale"/> x the camera resolution, or the camera's own texture when off.</summary>
        public Texture Output => _enabled && _result != null ? _result : _thermal.Temperature;

        public bool Enabled
        {
            get => _enabled;
            set
            {
                _enabled = value;
                PlayerPrefs.SetInt(Key, value ? 1 : 0);
                PlayerPrefs.Save();
                // A frozen frame gets no new FrameReady, so sharpen it now.
                if (value && _thermal.FrameCount > 0) Process();
            }
        }

        public string Label => _enabled ? $"Sharpen: {Strength:F0}" : "Sharpen: off";

        /// <summary>Subscribe before anything that reads <see cref="Output"/> on FrameReady.</summary>
        public void Init(ThermalCamera thermal)
        {
            _thermal = thermal;
            _enabled = PlayerPrefs.GetInt(Key, 1) == 1;
            _shader = Resources.Load<ComputeShader>("P2ProSharpen");
            _upsample = _shader.FindKernel("Upsample");
            _blurX = _shader.FindKernel("BlurX");
            _sharpen = _shader.FindKernel("Sharpen");
            int w = ThermalCamera.Width * Upscale, h = ThermalCamera.Height * Upscale;
            _up = Create("Thermal upsampled", w, h);
            _tmp = Create("Thermal blur", w, h);
            _result = Create("Thermal sharpened", w, h);

            _shader.SetTexture(_upsample, UpId, _up);
            _shader.SetTexture(_blurX, UpReadId, _up);
            _shader.SetTexture(_blurX, TmpId, _tmp);
            _shader.SetTexture(_sharpen, UpReadId, _up);
            _shader.SetTexture(_sharpen, TmpReadId, _tmp);
            _shader.SetTexture(_sharpen, ResultId, _result);
            _shader.SetInts(SourceSizeId, ThermalCamera.Width, ThermalCamera.Height);
            _shader.SetInts(SizeId, w, h);
            _shader.SetFloat(SigmaId, Sigma);
            _shader.SetInt(RadiusId, 4);         // OpenCV's kernel for sigma 1.2 is 9 taps
            _shader.SetFloat(StrengthId, Strength);
            _thermal.FrameReady += OnFrame;
        }

        static RenderTexture Create(string name, int w, int h)
        {
            var rt = new RenderTexture(w, h, 0, GraphicsFormat.R32_SFloat)
            {
                name = name,
                enableRandomWrite = true,
                wrapMode = TextureWrapMode.Clamp,
                filterMode = SystemInfo.IsFormatSupported(GraphicsFormat.R32_SFloat, GraphicsFormatUsage.Linear)
                    ? FilterMode.Bilinear : FilterMode.Point,
            };
            rt.Create();
            return rt;
        }

        void OnFrame()
        {
            if (_enabled) Process();
        }

        void Process()
        {
            _shader.SetTexture(_upsample, SourceId, _thermal.Temperature);
            int gx = (_up.width + 7) / 8, gy = (_up.height + 7) / 8;
            _shader.Dispatch(_upsample, gx, gy, 1);
            _shader.Dispatch(_blurX, gx, gy, 1);
            _shader.Dispatch(_sharpen, gx, gy, 1);
        }

        void OnDestroy()
        {
            if (_thermal != null) _thermal.FrameReady -= OnFrame;
            if (_up != null) _up.Release();
            if (_tmp != null) _tmp.Release();
            if (_result != null) _result.Release();
        }
    }
}
