using System;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// The thermal image as a flat picture, without projecting it on the room: a grid at the
    /// measured scene distance in front of the RGB camera, each vertex placed through the lens
    /// model (so its distortion shows) or the alignment mapping at that distance, like the
    /// overlay during calibration. The camera pose is the one at the
    /// frame's exposure, so the picture stays on the scene while the head turns.
    /// Style (blend, opacity, filter) from <see cref="OverlayStyle"/>.
    /// </summary>
    public class ThermalFlat : MonoBehaviour
    {
        static readonly int MainTexId = Shader.PropertyToID("_MainTex");
        static readonly int PaletteId = Shader.PropertyToID("_Palette");
        static readonly int MinId = Shader.PropertyToID("_Min");
        static readonly int MaxId = Shader.PropertyToID("_Max");
        const int GridX = 16, GridY = 12;     // cells

        ThermalPanel _panel;
        ThermalAlignment _alignment;
        LatencyEstimator _latency;
        OverlayStyle _style;
        Func<float> _distance;
        Material _material;
        Mesh _mesh;
        MeshRenderer _renderer;
        readonly Vector3[] _vertices = new Vector3[(GridX + 1) * (GridY + 1)];
        readonly Vector2[] _uvs = new Vector2[(GridX + 1) * (GridY + 1)];
        bool _visible;
        bool _placed;

        public bool Visible
        {
            get => _visible;
            set
            {
                _visible = value;
                if (!value) _renderer.enabled = _placed = false;
            }
        }

        /// <param name="distance">Scene distance in metres to put the picture at.</param>
        public void Init(ThermalPanel panel, ThermalAlignment alignment, LatencyEstimator latency, OverlayStyle style,
            Func<float> distance)
        {
            _panel = panel;
            _alignment = alignment;
            _latency = latency;
            _style = style;
            _distance = distance;
            _material = new Material(Resources.Load<Shader>("P2ProThermalFlat"));
            _mesh = new Mesh { name = "Thermal flat" };
            var triangles = new int[GridX * GridY * 6];
            for (int y = 0, t = 0; y <= GridY; y++)
            for (int x = 0; x <= GridX; x++)
            {
                int i = y * (GridX + 1) + x;
                _uvs[i] = new Vector2((float)x / GridX, (float)y / GridY);
                if (x == GridX || y == GridY) continue;
                triangles[t++] = i; triangles[t++] = i + GridX + 1; triangles[t++] = i + 1;
                triangles[t++] = i + 1; triangles[t++] = i + GridX + 1; triangles[t++] = i + GridX + 2;
            }
            _mesh.vertices = _vertices;
            _mesh.uv = _uvs;
            _mesh.triangles = triangles;

            var go = new GameObject("Thermal flat");
            go.AddComponent<MeshFilter>().sharedMesh = _mesh;
            _renderer = go.AddComponent<MeshRenderer>();
            _renderer.sharedMaterial = _material;
            _renderer.enabled = false;
        }

        void LateUpdate()
        {
            if (!_visible) return;
            _placed = Place();
            _renderer.enabled = _placed;
            if (!_placed) return;
            _style.Apply(_material);
            _material.SetTexture(MainTexId, _panel.Temperature);
            _material.SetTexture(PaletteId, _panel.Palette);
            _material.SetFloat(MinId, _panel.RangeMin);
            _material.SetFloat(MaxId, _panel.RangeMax);
        }

        bool Place()
        {
            var source = _panel.Rgb.Source;
            if (source == null || !source.IsPlaying || _panel.Thermal.FrameCount == 0) return false;
            float distance = _distance();
            var pose = RgbProjection.FramePose(_panel, _latency);
            for (int i = 0; i < _vertices.Length; i++)
            {
                if (!_alignment.ThermalUvToRgb(_uvs[i], distance, source, out var rgb)) return false;
                _vertices[i] = pose.position + pose.rotation * rgb;
            }
            _mesh.vertices = _vertices;
            _mesh.RecalculateBounds();
            return true;
        }

        void OnDestroy()
        {
            if (_renderer != null) Destroy(_renderer.gameObject);
            Destroy(_mesh);
            Destroy(_material);
        }
    }
}
