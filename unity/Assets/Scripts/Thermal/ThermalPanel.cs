using System;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// World-space debug view: the live P2 Pro image coloured by a palette (left) and the headset
    /// RGB camera image (right), optionally with the thermal image laid over the RGB one, a row of
    /// pointer buttons above and a status text below. Shown in front of the head shortly after
    /// start. Also sets up the RGB camera, the thermal/RGB frame pairing and Align mode.
    /// Menu (left controller) hides / shows the panel and brings it back in front of you; right
    /// stick click = next palette, left stick click = range Scene / Frame / Wide.
    /// </summary>
    public class ThermalPanel : MonoBehaviour
    {
        enum RangeMode { Frame, Room, Wide, Scene }
        static readonly string[] RangeNames = { "Frame", "Room 15-35", "Wide", "Scene" };
        static readonly RangeMode[] QuickRanges = { RangeMode.Scene, RangeMode.Frame, RangeMode.Wide };   // left stick

        const float PanelWidth = 0.4f;
        const float Gap = 0.02f;
        const float Distance = 0.7f;
        const float DropBelowEyes = 0.1f;
        const float ShowAfterSeconds = 1.5f;
        const float OverlayAlpha = 0.5f;

        static readonly int MainTexId = Shader.PropertyToID("_MainTex");
        static readonly int PaletteId = Shader.PropertyToID("_Palette");
        static readonly int MinId = Shader.PropertyToID("_Min");
        static readonly int MaxId = Shader.PropertyToID("_Max");
        static readonly int AlphaId = Shader.PropertyToID("_Alpha");

        public ThermalCamera Thermal { get; private set; }
        public ThermalFlatField FlatField { get; private set; }
        public ThermalSharpening Sharpening { get; private set; }
        /// <summary>°C to display: the camera frame, sharpened when that is on.</summary>
        public Texture Temperature => Sharpening.Output;
        public HeadsetRgbCamera Rgb { get; private set; }
        public bool Frozen => _frozenRgb != null;
        public Texture Palette => _palettes[_palette];
        public string PaletteName => Palettes.Names[_palette];
        public string RangeName => RangeNames[(int)_range];
        /// <summary>
        /// Range of the temperatures kept in the room's thermal map, for the Scene range (the map
        /// sets this); null while there is none, and Scene then falls back to Frame.
        /// </summary>
        public Func<Vector2?> SceneRange { get; set; }
        public float RangeMin { get; private set; } = 15;
        public float RangeMax { get; private set; } = 35;

        Material _material;
        Material _rgbMaterial;
        Material _overlayMaterial;
        Transform _panel;
        Transform _thermalQuad;
        Transform _rgbQuad;
        Vector3 _thermalScale;
        float _overlayRoll;         // roll of the mapping on show, 0 without
        float? _thermalRoll;        // display roll asked for by the points wizard
        Transform _overlay;
        Vector3 _rgbCentre;
        float _aspect;
        Texture2D[] _palettes;
        int _palette;
        RangeMode _range = RangeMode.Scene;     // falls back to Frame until the map has data
        float _autoMin = float.NaN;
        float _autoMax = float.NaN;
        bool _placed;
        bool _thermalShown;
        RenderTexture _frozenRgb;

        void Start()
        {
            Thermal = gameObject.AddComponent<ThermalCamera>();
            FlatField = gameObject.AddComponent<ThermalFlatField>();
            FlatField.Init();
            Sharpening = gameObject.AddComponent<ThermalSharpening>();
            Sharpening.Init(Thermal);
            Thermal.FrameReady += OnFrame;
            Rgb = gameObject.AddComponent<HeadsetRgbCamera>();
            gameObject.AddComponent<FramePairing>().Init(Thermal, Rgb);

            _palettes = new Texture2D[Palettes.Names.Length];
            for (int i = 0; i < _palettes.Length; i++) _palettes[i] = Palettes.Build(i);

            _material = new Material(Resources.Load<Shader>("P2ProThermal"));
            _material.SetTexture(PaletteId, _palettes[_palette]);
            _overlayMaterial = new Material(_material) { renderQueue = 3000 };
            _overlayMaterial.SetFloat(AlphaId, OverlayAlpha);

            _rgbMaterial = new Material(Resources.Load<Shader>("P2ProRgb"));

            var root = new GameObject("Thermal debug view");
            root.SetActive(false);
            _panel = root.transform;
            _aspect = (float)ThermalCamera.Height / ThermalCamera.Width;  // 3:4, RGB is 1280x960 too
            _thermalQuad = AddQuad("Thermal", _material, -(PanelWidth + Gap) / 2);
            _rgbQuad = AddQuad("RGB", _rgbMaterial, (PanelWidth + Gap) / 2);
            _thermalScale = _thermalQuad.localScale;
            _rgbCentre = _rgbQuad.localPosition;
            _overlay = AddQuad("Thermal over RGB", _overlayMaterial, 0);
            _overlay.gameObject.SetActive(false);

            var text = PanelText.Create(_panel, new Vector3(-PanelWidth - Gap / 2, -PanelWidth * _aspect / 2 - Gap, 0), "Status");
            var buttons = gameObject.AddComponent<ButtonBar>();
            buttons.Init(_panel, new Vector3(0, PanelWidth * _aspect / 2 + 0.15f, 0), 2 * PanelWidth + Gap);  // up to three rows
            gameObject.AddComponent<AlignMode>().Init(this, text, buttons);
        }

        Transform AddQuad(string name, Material material, float x)
        {
            var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            quad.name = name;
            Destroy(quad.GetComponent<Collider>());
            quad.GetComponent<MeshRenderer>().sharedMaterial = material;
            quad.transform.SetParent(_panel, false);
            quad.transform.localPosition = new Vector3(x, 0, 0);
            quad.transform.localScale = new Vector3(PanelWidth, PanelWidth * _aspect, 1);
            return quad.transform;
        }

        /// <summary>
        /// Lays the thermal image over the RGB one where <paramref name="m"/> puts it, and turns
        /// the stand-alone thermal image by the same roll; null hides the overlay.
        /// Display only: temperatures stay in the camera's own image frame.
        /// </summary>
        public void ShowOverlay(ThermalAlignment.Mapping? m)
        {
            _overlay.gameObject.SetActive(m.HasValue);
            _overlayRoll = m.HasValue ? m.Value.RollDeg : 0;
            TurnThermalQuad();
            if (!m.HasValue) return;
            var v = m.Value;
            // The RGB quad is PanelWidth wide, so RGB-width units scale by PanelWidth.
            _overlay.localPosition = _rgbCentre +
                new Vector3((v.U - 0.5f) * PanelWidth, (v.V - 0.5f) * PanelWidth * _aspect, -0.002f);
            _overlay.localRotation = Quaternion.Euler(0, 0, v.RollDeg);
            _overlay.localScale = new Vector3(v.Scale * PanelWidth, v.Scale * PanelWidth * _aspect, 1);
        }

        /// <summary>
        /// Roll (degrees) to show the stand-alone thermal image at instead of the mapping's, or
        /// null for the mapping's. Display only, as <see cref="ShowOverlay"/>; <see cref="HitImage"/>
        /// and <see cref="ImagePoint"/> follow it.
        /// </summary>
        public float? ThermalRoll
        {
            get => _thermalRoll;
            set { _thermalRoll = value; TurnThermalQuad(); }
        }

        // Turned and shrunk to stay within its own place on the panel.
        void TurnThermalQuad()
        {
            float roll = _thermalRoll ?? _overlayRoll;
            float c = Mathf.Abs(Mathf.Cos(roll * Mathf.Deg2Rad)), s = Mathf.Abs(Mathf.Sin(roll * Mathf.Deg2Rad));
            float fit = Mathf.Min(1, Mathf.Min(1 / (c + _aspect * s), _aspect / (s + _aspect * c)));
            _thermalQuad.localRotation = Quaternion.Euler(0, 0, roll);
            _thermalQuad.localScale = _thermalScale * fit;
        }

        /// <summary>The panel, for things placed on it.</summary>
        public Transform Root => _panel;

        /// <summary>
        /// Where a world ray meets the thermal (left) or the RGB (right) image, as that image's
        /// texture coordinate (0..1, y up); false when it misses both or the panel is hidden.
        /// </summary>
        public bool HitImage(Ray ray, out bool thermal, out Vector2 uv)
        {
            thermal = false;
            uv = default;
            if (_panel == null || !_panel.gameObject.activeInHierarchy) return false;
            foreach (bool t in new[] { true, false })
            {
                var quad = t ? _thermalQuad : _rgbQuad;
                if (!new Plane(quad.forward, quad.position).Raycast(ray, out float d)) continue;
                var local = quad.InverseTransformPoint(ray.GetPoint(d));
                if (Mathf.Abs(local.x) > 0.5f || Mathf.Abs(local.y) > 0.5f) continue;
                thermal = t;
                uv = new Vector2(local.x + 0.5f, local.y + 0.5f);
                return true;
            }
            return false;
        }

        /// <summary>World point of a texture coordinate on the thermal or RGB image, just in front of it.</summary>
        public Vector3 ImagePoint(bool thermal, Vector2 uv)
        {
            var quad = thermal ? _thermalQuad : _rgbQuad;
            return quad.TransformPoint(new Vector3(uv.x - 0.5f, uv.y - 0.5f, 0)) - quad.forward * 0.003f;
        }

        /// <summary>
        /// Holds the current RGB image and temperatures on screen until <see cref="Unfreeze"/>;
        /// the thermal camera is paused, so <see cref="ThermalCamera.Celsius"/> keeps the frozen frame.
        /// </summary>
        public bool Freeze()
        {
            var rgb = Rgb.Texture;
            if (rgb == null || Thermal.FrameCount == 0) return false;
            _frozenRgb = new RenderTexture(rgb.width, rgb.height, 0, RenderTextureFormat.ARGB32) { name = "Frozen RGB" };
            Graphics.Blit(rgb, _frozenRgb);
            Thermal.Paused = true;
            SetTextures();
            return true;
        }

        public void Unfreeze()
        {
            if (!Frozen) return;
            Destroy(_frozenRgb);
            _frozenRgb = null;
            Thermal.Paused = false;
            SetTextures();
        }

        public void NextPalette()
        {
            _palette = (_palette + 1) % _palettes.Length;
            _material.SetTexture(PaletteId, _palettes[_palette]);
            _overlayMaterial.SetTexture(PaletteId, _palettes[_palette]);
            Debug.Log("[P2Pro] palette " + Palettes.Names[_palette]);
        }

        public void NextRange() => SetRange((RangeMode)(((int)_range + 1) % RangeNames.Length));

        // From a range missing in the quick order (Room) the next is the first.
        void NextQuickRange() => SetRange(QuickRanges[(Array.IndexOf(QuickRanges, _range) + 1) % QuickRanges.Length]);

        void SetRange(RangeMode range)
        {
            _range = range;
            _autoMin = _autoMax = float.NaN;
            Debug.Log("[P2Pro] range " + _range);
        }

        void SetTextures()
        {
            _material.SetTexture(MainTexId, Temperature);
            _overlayMaterial.SetTexture(MainTexId, Temperature);
            _rgbMaterial.SetTexture(MainTexId, Frozen ? _frozenRgb : Rgb.Texture);
        }

        void Update()
        {
            if (!_placed && Time.timeSinceLevelLoad > ShowAfterSeconds && Camera.main != null)
            {
                _panel.gameObject.SetActive(true);
                PlaceInFrontOfHead();
                _placed = true;
            }

            if (_thermalShown && _material.GetTexture(MainTexId) != Temperature) SetTextures();
            var rgbTexture = Rgb.Texture;
            if (!Frozen && rgbTexture != null && _rgbMaterial.GetTexture(MainTexId) != rgbTexture)
                _rgbMaterial.SetTexture(MainTexId, rgbTexture);

            if (_placed && OVRInput.GetDown(OVRInput.Button.Start))
            {
                bool show = !_panel.gameObject.activeSelf;
                _panel.gameObject.SetActive(show);
                if (show) PlaceInFrontOfHead();
            }
            if (OVRInput.GetDown(OVRInput.Button.SecondaryThumbstick)) NextPalette();
            if (OVRInput.GetDown(OVRInput.Button.PrimaryThumbstick)) NextQuickRange();
        }

        void OnFrame()
        {
            if (!_thermalShown)
            {
                _thermalShown = true;
                SetTextures();
            }
            if (Frozen) return;

            float lo, hi;
            switch (_range)
            {
                case RangeMode.Room: lo = 15; hi = 35; break;
                case RangeMode.Wide: lo = -20; hi = 150; break;
                default:
                    // Frame: this frame's min / max; Scene: the room map's percentiles. Smooth the
                    // endpoints so the palette glides instead of jumping.
                    var target = _range == RangeMode.Scene && SceneRange?.Invoke() is { } scene
                        ? scene : new Vector2(Thermal.Min, Thermal.Max);
                    _autoMin = float.IsNaN(_autoMin) ? target.x : _autoMin + 0.2f * (target.x - _autoMin);
                    _autoMax = float.IsNaN(_autoMax) ? target.y : _autoMax + 0.2f * (target.y - _autoMax);
                    lo = _autoMin;
                    hi = Mathf.Max(_autoMax, _autoMin + 0.5f);
                    break;
            }
            RangeMin = lo;
            RangeMax = hi;
            _material.SetFloat(MinId, lo);
            _material.SetFloat(MaxId, hi);
            _overlayMaterial.SetFloat(MinId, lo);
            _overlayMaterial.SetFloat(MaxId, hi);
        }

        public void PlaceInFrontOfHead()
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
            if (Thermal != null) Thermal.FrameReady -= OnFrame;
            Unfreeze();
            if (_panel != null) Destroy(_panel.gameObject);
            Destroy(_material);
            Destroy(_rgbMaterial);
            Destroy(_overlayMaterial);
            foreach (var p in _palettes) Destroy(p);
        }
    }
}
