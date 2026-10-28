using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// World-space debug view: the live P2 Pro image coloured by a palette (left) and the headset
    /// RGB camera image (right). Placed in front of the head when the first thermal frame arrives.
    /// Also sets up the RGB camera and the thermal/RGB frame pairing.
    /// Controls: B = next palette, right thumbstick click = range mode, Y = bring the panel back
    /// in front of you.
    /// </summary>
    public class ThermalPanel : MonoBehaviour
    {
        enum RangeMode { Auto, Room, Wide }

        const float PanelWidth = 0.4f;
        const float Gap = 0.02f;
        const float Distance = 0.7f;
        const float DropBelowEyes = 0.1f;

        static readonly int MainTexId = Shader.PropertyToID("_MainTex");
        static readonly int PaletteId = Shader.PropertyToID("_Palette");
        static readonly int MinId = Shader.PropertyToID("_Min");
        static readonly int MaxId = Shader.PropertyToID("_Max");

        static readonly int RgbTexId = Shader.PropertyToID("_MainTex");

        ThermalCamera _camera;
        HeadsetRgbCamera _rgb;
        Material _material;
        Material _rgbMaterial;
        Transform _panel;
        Texture2D[] _palettes;
        int _palette;
        RangeMode _range;
        float _autoMin = float.NaN;
        float _autoMax = float.NaN;
        bool _placed;

        void Start()
        {
            _camera = gameObject.AddComponent<ThermalCamera>();
            _camera.FrameReady += OnFrame;
            _rgb = gameObject.AddComponent<HeadsetRgbCamera>();
            gameObject.AddComponent<FramePairing>().Init(_camera, _rgb);

            _palettes = new Texture2D[Palettes.Names.Length];
            for (int i = 0; i < _palettes.Length; i++) _palettes[i] = Palettes.Build(i);

            _material = new Material(Resources.Load<Shader>("P2ProThermal"));
            _material.SetTexture(PaletteId, _palettes[_palette]);

            _rgbMaterial = new Material(Resources.Load<Shader>("P2ProRgb"));

            var root = new GameObject("Thermal debug view");
            root.SetActive(false);
            _panel = root.transform;
            float aspect = (float)ThermalCamera.Height / ThermalCamera.Width;  // 3:4, RGB is 1280x960 too
            AddQuad("Thermal", _material, -(PanelWidth + Gap) / 2, aspect);
            AddQuad("RGB", _rgbMaterial, (PanelWidth + Gap) / 2, aspect);
        }

        void AddQuad(string name, Material material, float x, float aspect)
        {
            var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            quad.name = name;
            Destroy(quad.GetComponent<Collider>());
            quad.GetComponent<MeshRenderer>().sharedMaterial = material;
            quad.transform.SetParent(_panel, false);
            quad.transform.localPosition = new Vector3(x, 0, 0);
            quad.transform.localScale = new Vector3(PanelWidth, PanelWidth * aspect, 1);
        }

        void Update()
        {
            var rgbTexture = _rgb.Texture;
            if (rgbTexture != null && _rgbMaterial.GetTexture(RgbTexId) != rgbTexture)
                _rgbMaterial.SetTexture(RgbTexId, rgbTexture);

            if (OVRInput.GetDown(OVRInput.Button.Four)) PlaceInFrontOfHead();
            if (OVRInput.GetDown(OVRInput.Button.Two))
            {
                _palette = (_palette + 1) % _palettes.Length;
                _material.SetTexture(PaletteId, _palettes[_palette]);
                Debug.Log("[P2Pro] palette " + Palettes.Names[_palette]);
            }
            if (OVRInput.GetDown(OVRInput.Button.SecondaryThumbstick))
            {
                _range = (RangeMode)(((int)_range + 1) % 3);
                _autoMin = _autoMax = float.NaN;
                Debug.Log("[P2Pro] range " + _range);
            }
        }

        void OnFrame()
        {
            if (!_placed)
            {
                _material.SetTexture(MainTexId, _camera.Temperature);
                _panel.gameObject.SetActive(true);
                PlaceInFrontOfHead();
                _placed = true;
            }

            float lo, hi;
            switch (_range)
            {
                case RangeMode.Room: lo = 15; hi = 35; break;
                case RangeMode.Wide: lo = -20; hi = 150; break;
                default:
                    // Smooth the endpoints so the palette does not flicker with single hot pixels.
                    _autoMin = float.IsNaN(_autoMin) ? _camera.Min : _autoMin + 0.2f * (_camera.Min - _autoMin);
                    _autoMax = float.IsNaN(_autoMax) ? _camera.Max : _autoMax + 0.2f * (_camera.Max - _autoMax);
                    lo = _autoMin;
                    hi = Mathf.Max(_autoMax, _autoMin + 0.5f);
                    break;
            }
            _material.SetFloat(MinId, lo);
            _material.SetFloat(MaxId, hi);
        }

        void PlaceInFrontOfHead()
        {
            var head = Camera.main != null ? Camera.main.transform : null;
            if (head == null) return;
            var forward = Vector3.ProjectOnPlane(head.forward, Vector3.up);
            if (forward.sqrMagnitude < 1e-4f) forward = Vector3.forward;
            forward.Normalize();
            _panel.SetPositionAndRotation(
                head.position + forward * Distance + Vector3.down * DropBelowEyes,
                Quaternion.LookRotation(forward));
        }

        void OnDestroy()
        {
            if (_camera != null) _camera.FrameReady -= OnFrame;
            if (_panel != null) Destroy(_panel.gameObject);
            Destroy(_material);
            Destroy(_rgbMaterial);
            foreach (var p in _palettes) Destroy(p);
        }
    }
}
