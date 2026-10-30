using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace P2Pro
{
    /// <summary>
    /// Evens out the P2 Pro's sensor where temperatures land on the room, never on the live
    /// image (P2ProFlatField.compute): the sensor reads a few tenths of a degree differently
    /// towards its edges, which showed as seams on the painted room. A 32 x 24 grid of offsets
    /// is learnt from the room itself: wherever a surface has a reliable running average in
    /// <see cref="ThermalMap"/>, the frame's difference from it is collected per grid cell by
    /// the bake, and each frame the offsets move 2 % towards it. The image centre stays at 0, so
    /// readings keep the camera's own calibration. The bake subtracts the offsets from what it
    /// writes into the map and P2Pro/ThermalMesh from what it reads out of stored keyframes
    /// (P2ProFlatOffset.hlsl); the panel, Live and Flat show the camera's frame as it is.
    /// Off by default; on/off is remembered between launches. Offsets decay 0.2 % a frame and
    /// stay within 1 C (P2ProFlatField.compute), so they cannot drift into bright image edges.
    /// </summary>
    public class ThermalFlatField : MonoBehaviour
    {
        public const int GridX = 32, GridY = 24;
        const float Rate = 0.02f;
        const float Decay = 0.002f;
        const float LogInterval = 10;
        const string Key = "p2pro.flatField.v2";     // v1 defaulted on

        static readonly int OffsetsId = Shader.PropertyToID("_Offsets");
        static readonly int FlatOffsetsId = Shader.PropertyToID("_FlatOffsets");
        static readonly int FlatOnId = Shader.PropertyToID("_FlatOn");
        static readonly int FlatSumId = Shader.PropertyToID("_FlatSum");
        static readonly int FlatCountId = Shader.PropertyToID("_FlatCount");
        static readonly int RateId = Shader.PropertyToID("_Rate");
        static readonly int DecayId = Shader.PropertyToID("_Decay");

        ComputeShader _shader;
        int _update, _reset;
        ComputeBuffer _offsets, _sum, _count;
        bool _enabled;
        bool _readback;
        float _nextLog;

        /// <summary>Residual sums the bake adds to (int millidegrees and counts per cell).</summary>
        public ComputeBuffer Sum => _sum;
        public ComputeBuffer Count => _count;

        /// <summary>Spread of the learnt offsets (max - min, °C), refreshed every few seconds.</summary>
        public float Spread { get; private set; }

        public bool Enabled
        {
            get => _enabled;
            set
            {
                _enabled = value;
                PlayerPrefs.SetInt(Key, value ? 1 : 0);
                PlayerPrefs.Save();
                if (value) Reset();
            }
        }

        public string Label => _enabled ? $"Even: {Spread:F1} C" : "Even: off";

        public void Init()
        {
            _enabled = PlayerPrefs.GetInt(Key, 0) == 1;
            _shader = Resources.Load<ComputeShader>("P2ProFlatField");
            _update = _shader.FindKernel("Update");
            _reset = _shader.FindKernel("Reset");
            int cells = GridX * GridY;
            _offsets = new ComputeBuffer(cells, 4);
            _sum = new ComputeBuffer(cells, 4);
            _count = new ComputeBuffer(cells, 4);
            foreach (int kernel in new[] { _update, _reset })
            {
                _shader.SetBuffer(kernel, OffsetsId, _offsets);
                _shader.SetBuffer(kernel, FlatSumId, _sum);
                _shader.SetBuffer(kernel, FlatCountId, _count);
            }
            _shader.SetFloat(RateId, Rate);
            _shader.SetFloat(DecayId, Decay);
            Reset();
        }

        /// <summary>Forgets the learnt offsets.</summary>
        public void Reset()
        {
            _shader.Dispatch(_reset, 1, 1, 1);
            Spread = 0;
        }

        /// <summary>Folds the residuals the bake collected for this frame into the offsets.</summary>
        public void Learn()
        {
            if (_enabled) _shader.Dispatch(_update, 1, 1, 1);
        }

        /// <summary>Offsets for P2ProFlatOffset.hlsl (zero effect while off).</summary>
        public void Apply(ComputeShader shader, int kernel)
        {
            shader.SetBuffer(kernel, FlatOffsetsId, _offsets);
            shader.SetFloat(FlatOnId, _enabled ? 1 : 0);
        }

        public void Apply(Material material)
        {
            material.SetBuffer(FlatOffsetsId, _offsets);
            material.SetFloat(FlatOnId, _enabled ? 1 : 0);
        }

        void Update()
        {
            if (!_enabled || _readback || Time.unscaledTime < _nextLog) return;
            _nextLog = Time.unscaledTime + LogInterval;
            _readback = true;
            AsyncGPUReadback.Request(_offsets, request =>
            {
                _readback = false;
                if (request.hasError || _offsets == null) return;
                var data = request.GetData<float>();
                float min = float.MaxValue, max = float.MinValue;
                foreach (float o in data) { min = Mathf.Min(min, o); max = Mathf.Max(max, o); }
                Spread = max - min;
                // Left, centre and right of the middle row: the shading that painted seams.
                int row = GridY / 2 * GridX;
                Debug.Log($"[P2Pro] flat field: spread {Spread:F2} C, middle row left {data[row]:F2} centre {data[row + GridX / 2]:F2} right {data[row + GridX - 1]:F2} C");
            });
        }

        void OnDestroy()
        {
            _offsets?.Release();
            _sum?.Release();
            _count?.Release();
            _offsets = _sum = _count = null;
        }
    }
}
