using System;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Heads-up display in the lower left of the view, on a dark rounded card: what the app is
    /// doing (rich text from <see cref="AlignMode"/>) and the temperature band selector:
    /// <list type="bullet">
    /// <item>the palette bar over the current colour range, with two handles for the band
    /// (<see cref="OverlayStyle.BandLow"/> .. <see cref="OverlayStyle.BandHigh"/>); the part that
    /// is not coloured is dimmed;</item>
    /// <item>a switch: All (every temperature coloured), Inside (only the band), Outside (all but
    /// the band).</item>
    /// </list>
    /// Point a controller at it and pull the trigger: the nearer handle follows the ray while
    /// the trigger is held; a press on the switch picks that mode. Dragging a handle in All turns
    /// to Inside. The card follows the head with a short lag, and holds still while a ray is on
    /// it. Shown / hidden with B (<see cref="ControllerGuide"/>; remembered between launches).
    /// </summary>
    public class Hud : MonoBehaviour
    {
        const string Key = "p2pro.hud";
        static readonly Vector3 Offset = new(-0.235f, -0.04f, 0.55f);      // card's top left, from the eyes, metres
        const float Width = 0.22f;
        const float Pad = 0.009f;
        const float LineHeight = 0.0085f;
        const float BarHeight = 0.011f;
        const float HandleWidth = 0.0035f, HandleHeight = 0.019f;
        const float SwitchHeight = 0.013f, SwitchGap = 0.003f;
        const float LabelHeight = 0.0072f;
        const float FollowRate = 8;             // 1/s
        const float TextInterval = 0.25f;
        const float RayLength = 2;

        static readonly Color Card = new(0.06f, 0.07f, 0.09f, 0.82f);
        static readonly Color CardBorder = new(1, 1, 1, 0.08f);
        static readonly Color Dim = new(0.03f, 0.03f, 0.04f, 0.78f);
        static readonly Color HandleColour = new(0.96f, 0.97f, 1, 1);
        static readonly Color HandleHover = new(1, 0.78f, 0.2f, 1);
        static readonly Color SwitchOff = new(1, 1, 1, 0.07f);
        static readonly Color SwitchOn = new(0.25f, 0.62f, 1, 0.9f);
        static readonly Color SwitchHover = new(1, 1, 1, 0.16f);
        static readonly Color Muted = new(0.62f, 0.66f, 0.74f);
        static readonly string[] SwitchNames = { "All", "Inside", "Outside" };

        ThermalPanel _panel;
        OverlayStyle _style;
        Func<string> _status;
        Transform _root, _head, _trackingSpace;
        PanelText _text, _min, _max, _lowLabel, _highLabel;
        UiRect _card, _bar, _dimLeft, _dimMiddle, _dimRight, _low, _high;
        readonly UiRect[] _switch = new UiRect[3];
        readonly PanelText[] _switchText = new PanelText[3];
        LineRenderer _ray;
        Material _rayMaterial;
        bool _visible;
        bool _placed;
        float _nextText;
        float _barY;

        // Dragging a handle: which (0 low, 1 high), with which controller, over which span.
        int _dragHandle = -1;
        OVRInput.Controller _dragController;
        float _dragMin, _dragMax;
        Collider _hovered;
        int _nearer;                            // handle nearer the ray on the bar

        public void Init(ThermalPanel panel, OverlayStyle style, Func<string> status)
        {
            _panel = panel;
            _style = style;
            _status = status;
            _root = new GameObject("HUD").transform;

            _card = UiRect.Create(_root, "HUD card", Vector3.zero, new Vector2(Width, 0.1f), Card, 0.012f, onTop: true);
            _card.SetBorder(CardBorder, 0.0008f);
            _text = PanelText.Create(_root, new Vector3(Pad, -Pad, -0.001f), "HUD text", TextAnchor.UpperLeft, LineHeight);
            _text.AlwaysOnTop();

            float barWidth = Width - 2 * Pad;
            _bar = UiRect.Create(_root, "HUD palette", Vector3.zero, new Vector2(barWidth, BarHeight), Color.white, 0.003f, onTop: true, collider: true);
            _dimLeft = UiRect.Create(_root, "HUD dim left", Vector3.zero, Vector2.one * 0.001f, Dim, 0.002f, onTop: true);
            _dimMiddle = UiRect.Create(_root, "HUD dim middle", Vector3.zero, Vector2.one * 0.001f, Dim, 0.002f, onTop: true);
            _dimRight = UiRect.Create(_root, "HUD dim right", Vector3.zero, Vector2.one * 0.001f, Dim, 0.002f, onTop: true);
            _low = UiRect.Create(_root, "HUD band low", Vector3.zero, new Vector2(HandleWidth, HandleHeight), HandleColour, 0.0015f, onTop: true);
            _high = UiRect.Create(_root, "HUD band high", Vector3.zero, new Vector2(HandleWidth, HandleHeight), HandleColour, 0.0015f, onTop: true);
            _lowLabel = Label("HUD band low label", TextAnchor.LowerCenter);
            _highLabel = Label("HUD band high label", TextAnchor.LowerCenter);
            _min = Label("HUD min", TextAnchor.UpperLeft);
            _max = Label("HUD max", TextAnchor.UpperRight);
            float w = (barWidth - 2 * SwitchGap) / 3;
            for (int i = 0; i < 3; i++)
            {
                _switch[i] = UiRect.Create(_root, "HUD switch " + SwitchNames[i], Vector3.zero, new Vector2(w, SwitchHeight), SwitchOff, 0.004f, onTop: true, collider: true);
                _switchText[i] = Label("HUD switch label", TextAnchor.MiddleCenter);
                _switchText[i].Text = SwitchNames[i];
                _switch[i].Order = 1;
            }
            _bar.Order = 1;
            _dimLeft.Order = _dimMiddle.Order = _dimRight.Order = 2;
            _low.Order = _high.Order = 3;

            _rayMaterial = new Material(Resources.Load<Shader>("P2ProText")) { mainTexture = Texture2D.whiteTexture };
            _rayMaterial.SetFloat("_ZTest", (float)UnityEngine.Rendering.CompareFunction.Always);
            _rayMaterial.renderQueue = 4001;
            _ray = new GameObject("HUD pointer").AddComponent<LineRenderer>();
            _ray.sharedMaterial = _rayMaterial;
            _ray.widthMultiplier = 0.0015f;
            _ray.positionCount = 2;
            _ray.enabled = false;

            Visible = PlayerPrefs.GetInt(Key, 1) == 1;
        }

        PanelText Label(string name, TextAnchor anchor)
        {
            var label = PanelText.Create(_root, Vector3.zero, name, anchor, LabelHeight);
            label.AlwaysOnTop();
            return label;
        }

        public bool Visible
        {
            get => _visible;
            set
            {
                _visible = value;
                _root.gameObject.SetActive(value);
                if (!value) { _ray.enabled = false; _dragHandle = -1; }
                PlayerPrefs.SetInt(Key, value ? 1 : 0);
                PlayerPrefs.Save();
                _nextText = 0;
            }
        }

        void Update()
        {
            if (!_visible || !TryGetRig()) return;

            bool engaged = Point();
            if (!engaged || !_placed)
            {
                var target = new Pose(_head.position + _head.rotation * Offset, _head.rotation);
                float k = _placed ? 1 - Mathf.Exp(-FollowRate * Time.unscaledDeltaTime) : 1;
                _placed = true;
                _root.SetPositionAndRotation(Vector3.Lerp(_root.position, target.position, k),
                                             Quaternion.Slerp(_root.rotation, target.rotation, k));
            }

            if (Time.unscaledTime >= _nextText)
            {
                _nextText = Time.unscaledTime + TextInterval;
                _text.Text = _status();
                _bar.SetTexture(_panel.Palette);
                Layout();
            }
            PlaceBand();
        }

        // Card height from the text; bar, labels and switch below it.
        void Layout()
        {
            float textHeight = _text.Height;
            float y = -Pad - textHeight - 0.006f - LabelHeight - 0.002f;     // band labels sit above the bar
            _barY = y - BarHeight / 2;
            float barWidth = Width - 2 * Pad;
            _bar.transform.localPosition = new Vector3(Width / 2, _barY, -0.001f);
            _min.transform.localPosition = new Vector3(Pad, _barY - BarHeight / 2 - 0.0025f, -0.002f);
            _max.transform.localPosition = new Vector3(Width - Pad, _barY - BarHeight / 2 - 0.0025f, -0.002f);
            _min.Text = $"{_panel.RangeMin:F1}°";
            _max.Text = $"{_panel.RangeMax:F1}°";
            float switchY = _barY - BarHeight / 2 - 0.0025f - LabelHeight - 0.004f - SwitchHeight / 2;
            float w = (barWidth - 2 * SwitchGap) / 3;
            for (int i = 0; i < 3; i++)
            {
                var p = new Vector3(Pad + w / 2 + i * (w + SwitchGap), switchY, -0.001f);
                _switch[i].transform.localPosition = p;
                _switchText[i].transform.localPosition = p + new Vector3(0, 0, -0.001f);
            }
            float height = -(switchY - SwitchHeight / 2) + Pad;
            _card.SetSize(new Vector2(Width, height));
            _card.transform.localPosition = new Vector3(Width / 2, -height / 2, 0);
        }

        // Handles, their labels, the dimmed parts and the switch colours, every frame.
        void PlaceBand()
        {
            float barWidth = Width - 2 * Pad;
            float min = _dragHandle >= 0 ? _dragMin : _panel.RangeMin;
            float max = _dragHandle >= 0 ? _dragMax : _panel.RangeMax;
            float lo = Mathf.Clamp01(Mathf.InverseLerp(min, max, _style.BandLow));
            float hi = Mathf.Clamp01(Mathf.InverseLerp(min, max, _style.BandHigh));
            float x0 = Pad, xLo = Pad + lo * barWidth, xHi = Pad + hi * barWidth, x1 = Pad + barWidth;
            var filter = _style.Filter;

            Shade(_dimLeft, x0, xLo, filter == OverlayStyle.FilterKind.Inside);
            Shade(_dimRight, xHi, x1, filter == OverlayStyle.FilterKind.Inside);
            Shade(_dimMiddle, xLo, xHi, filter == OverlayStyle.FilterKind.Outside);

            bool active = filter != OverlayStyle.FilterKind.All;
            _low.transform.localPosition = new Vector3(xLo, _barY, -0.003f);
            _high.transform.localPosition = new Vector3(xHi, _barY, -0.003f);
            _low.Colour = Handle(0, active);
            _high.Colour = Handle(1, active);
            _lowLabel.transform.localPosition = new Vector3(xLo, _barY + HandleHeight / 2 + 0.001f, -0.003f);
            _highLabel.transform.localPosition = new Vector3(xHi, _barY + HandleHeight / 2 + 0.001f, -0.003f);
            _lowLabel.Text = $"{_style.BandLow:F1}";
            _highLabel.Text = $"{_style.BandHigh:F1}";
            var labelColour = active ? Color.white : Muted;
            _lowLabel.Colour = labelColour;
            _highLabel.Colour = labelColour;
            _min.Colour = Muted;
            _max.Colour = Muted;

            for (int i = 0; i < 3; i++)
            {
                bool on = (int)filter == i;
                _switch[i].Colour = on ? SwitchOn : _hovered == _switch[i].Collider ? SwitchHover : SwitchOff;
                _switchText[i].Colour = on ? Color.white : Muted;
            }
        }

        Color Handle(int handle, bool active) =>
            _dragHandle == handle || (_dragHandle < 0 && _hovered == _bar.Collider && _nearer == handle) ? HandleHover
            : active ? HandleColour : new Color(HandleColour.r, HandleColour.g, HandleColour.b, 0.45f);

        void Shade(UiRect rect, float from, float to, bool show)
        {
            float w = to - from;
            rect.gameObject.SetActive(show && w > 0.0005f);
            if (!rect.gameObject.activeSelf) return;
            rect.SetSize(new Vector2(w, BarHeight));
            rect.transform.localPosition = new Vector3(from + w / 2, _barY, -0.002f);
        }

        // Both controllers' rays against the card's controls; true while one is on it or dragging.
        bool Point()
        {
            _hovered = null;
            bool drawn = false;
            foreach (var controller in new[] { OVRInput.Controller.RTouch, OVRInput.Controller.LTouch })
            {
                if (!OVRInput.IsControllerConnected(controller)) continue;
                var position = OVRInput.GetLocalControllerPosition(controller);
                var rotation = OVRInput.GetLocalControllerRotation(controller);
                if (_trackingSpace != null)
                {
                    position = _trackingSpace.TransformPoint(position);
                    rotation = _trackingSpace.rotation * rotation;
                }
                var ray = new Ray(position, rotation * Vector3.forward);

                if (_dragHandle >= 0 && _dragController == controller)
                {
                    if (!OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger, controller)) _dragHandle = -1;
                    else if (Drag(ray, out var point))
                    {
                        Draw(position, point);
                        drawn = true;
                        continue;
                    }
                }

                if (!Physics.Raycast(ray, out var hit, RayLength) || !IsMine(hit.collider)) continue;
                _hovered = hit.collider;
                if (hit.collider == _bar.Collider) _nearer = Nearer(BarValue(hit.point));
                if (!drawn)
                {
                    Draw(position, hit.point);
                    drawn = true;
                }
                if (!OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, controller)) continue;
                for (int i = 0; i < 3; i++)
                    if (hit.collider == _switch[i].Collider)
                    {
                        _style.SetFilter((OverlayStyle.FilterKind)i, _panel.RangeMin, _panel.RangeMax);
                        Debug.Log("[P2Pro] HUD band: " + _style.FilterLabel);
                    }
                if (hit.collider == _bar.Collider)
                {
                    _dragMin = _panel.RangeMin;
                    _dragMax = _panel.RangeMax;
                    _dragController = controller;
                    _dragHandle = Nearer(BarValue(hit.point));
                    Drag(ray, out _);
                }
            }
            _ray.enabled = drawn;
            return _hovered != null || _dragHandle >= 0;
        }

        void Draw(Vector3 from, Vector3 to)
        {
            _ray.SetPosition(0, from);
            _ray.SetPosition(1, to);
        }

        bool IsMine(Collider c)
        {
            if (c == _bar.Collider) return true;
            foreach (var s in _switch) if (c == s.Collider) return true;
            return false;
        }

        int Nearer(float celsius) =>
            Mathf.Abs(celsius - _style.BandLow) <= Mathf.Abs(celsius - _style.BandHigh) ? 0 : 1;

        // °C under a world point on the bar, over the bar's current span.
        float BarValue(Vector3 world)
        {
            float x = _bar.transform.InverseTransformPoint(world).x + 0.5f;
            float min = _dragHandle >= 0 ? _dragMin : _panel.RangeMin;
            float max = _dragHandle >= 0 ? _dragMax : _panel.RangeMax;
            return Mathf.Lerp(min, max, Mathf.Clamp01(x));
        }

        // The held handle takes the °C where the ray crosses the bar's plane, to 0.1 °C.
        bool Drag(Ray ray, out Vector3 point)
        {
            var bar = _bar.transform;
            point = default;
            if (!new Plane(bar.forward, bar.position).Raycast(ray, out float d) || d > RayLength) return false;
            point = ray.GetPoint(d);
            float v = Mathf.Round(BarValue(point) * 10) / 10;
            if (_dragHandle == 0) _style.SetBand(Mathf.Min(v, _style.BandHigh - OverlayStyle.MinBand), _style.BandHigh);
            else _style.SetBand(_style.BandLow, Mathf.Max(v, _style.BandLow + OverlayStyle.MinBand));
            return true;
        }

        bool TryGetRig()
        {
            if (_head == null)
            {
                var rig = FindAnyObjectByType<OVRCameraRig>();
                if (rig == null) return false;
                _head = rig.centerEyeAnchor;
                _trackingSpace = rig.trackingSpace;
            }
            return _head != null;
        }

        void OnDestroy()
        {
            if (_root != null) Destroy(_root.gameObject);
            if (_ray != null) Destroy(_ray.gameObject);
            Destroy(_rayMaterial);
        }
    }
}
