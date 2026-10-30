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
        static readonly int RgbEdgeTexId = Shader.PropertyToID("_RgbEdgeTex");
        static readonly int WorldToRgbImageId = Shader.PropertyToID("_WorldToRgbImage");
        static readonly int RgbEdgesOnId = Shader.PropertyToID("_RgbEdgesOn");

        ThermalPanel _panel;
        ThermalAlignment _alignment;
        LatencyEstimator _latency;
        OverlayStyle _style;
        Material _material;
        bool _visible;

        /// <summary>RGB edges for the Edges look; none without.</summary>
        public RgbEdgeMap RgbEdges { get; set; }

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
            // RGB edges (Edges look) where the RGB image was taken, not where the thermal one was.
            var rgb = _panel.Rgb;
            bool rgbEdges = RgbEdges != null && RgbEdges.Texture != null && rgb.FrameCount > 0 && !_panel.Frozen;
            _material.SetFloat(RgbEdgesOnId, rgbEdges ? 1 : 0);
            if (rgbEdges)
            {
                _material.SetTexture(RgbEdgeTexId, RgbEdges.Texture);
                _material.SetMatrix(WorldToRgbImageId, RgbImagePose(rgb).inverse);
            }
        }

        /// <summary>World pose (as a matrix) of the RGB camera when its latest image was taken.</summary>
        public static Matrix4x4 RgbImagePose(HeadsetRgbCamera rgb)
        {
            var pose = rgb.CameraPose;          // PassthroughCameraAccess.GetCameraPose: world space
            return Matrix4x4.TRS(pose.position, pose.rotation, Vector3.one);
        }

        void OnDestroy()
        {
            GPUMeshRenderer.OverlayMaterials.Remove(_material);
            Destroy(_material);
        }
    }
}
