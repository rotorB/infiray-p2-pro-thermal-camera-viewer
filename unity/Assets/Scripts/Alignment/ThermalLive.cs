using Genesis.RoomScan;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Shows only the latest thermal frame on the room mesh (P2Pro/ThermalLive), keeping nothing:
    /// the live counterpart of <see cref="ThermalMap"/>.
    /// </summary>
    public class ThermalLive : MonoBehaviour
    {
        static readonly int MainTexId = Shader.PropertyToID("_MainTex");
        static readonly int PaletteId = Shader.PropertyToID("_Palette");
        static readonly int MinId = Shader.PropertyToID("_Min");
        static readonly int MaxId = Shader.PropertyToID("_Max");

        ThermalPanel _panel;
        ThermalAlignment _alignment;
        LatencyEstimator _latency;
        OverlayStyle _style;
        Material _material;
        bool _visible;

        public bool Visible
        {
            get => _visible;
            set
            {
                _visible = value;
                GPUMeshRenderer.OverlayMaterials.Remove(_material);
                if (value) GPUMeshRenderer.OverlayMaterials.Add(_material);
            }
        }

        public void Init(ThermalPanel panel, ThermalAlignment alignment, LatencyEstimator latency, OverlayStyle style)
        {
            _panel = panel;
            _style = style;
            _alignment = alignment;
            _latency = latency;
            _material = new Material(Resources.Load<Shader>("P2ProThermalLive"));
        }

        void LateUpdate()
        {
            if (!_visible) return;
            if (!RgbProjection.TryCreate(_panel, _alignment, _latency, out var projection))
            {
                // Nothing to show: push every point out of the thermal image.
                _material.SetVector(Shader.PropertyToID("_MapA"), new Vector4(10, 10, 1, 0));
                _material.SetVector(Shader.PropertyToID("_MapB"), Vector4.zero);
                return;
            }
            projection.Apply(_material);
            _style.Apply(_material);
            _material.SetTexture(MainTexId, _panel.Temperature);
            _material.SetTexture(PaletteId, _panel.Palette);
            _material.SetFloat(MinId, _panel.RangeMin);
            _material.SetFloat(MaxId, _panel.RangeMax);
        }

        void OnDestroy()
        {
            GPUMeshRenderer.OverlayMaterials.Remove(_material);
            Destroy(_material);
        }
    }
}
