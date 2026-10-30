using System;
using System.Collections.Generic;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Rows of buttons and sliders on the debug panel, used by pointing a controller at them and
    /// pulling its index trigger; a slider follows the ray while the trigger is held. Each
    /// controller draws a ray; the item under it lights up.
    /// The owner calls <see cref="Show"/> with the current items whenever they may change;
    /// the last row sits at the bar's position and the others stack upwards. A pressed button
    /// flashes; a disabled one is greyed out and, when pressed, says why in <see cref="Feedback"/>.
    /// </summary>
    public class ButtonBar : MonoBehaviour
    {
        public struct Button
        {
            public string Label;
            public Action OnClick;
            public bool Highlight;              // drawn in the accent colour (the main next step)
            public bool Enabled;
            public string Disabled;             // why it is disabled, shown when pressed anyway
            // Sliders only: value, its span along the slider, and its label.
            public Func<float> Get;
            public Action<float> Set;
            public float Min, Max;
            public Func<float, string> Format;

            public bool IsSlider => Set != null;

            public Button(string label, Action onClick, bool highlight = false, bool enabled = true, string disabled = null)
            {
                Label = label;
                OnClick = onClick;
                Highlight = highlight;
                Enabled = enabled;
                Disabled = disabled;
                Get = null;
                Set = null;
                Min = Max = 0;
                Format = null;
            }

            public static Button Slider(Func<float> get, Action<float> set, float min, float max, Func<float, string> format) =>
                new(null, null) { Get = get, Set = set, Min = min, Max = Mathf.Max(max, min + 0.01f), Format = format };
        }

        class Slot
        {
            public GameObject Root;
            public UiRect Background;
            public UiRect Fill, Handle;         // sliders only
            public PanelText Text;
            public Button Button;
            public float Width;
        }

        // A slider held by a controller's trigger; its span is kept from the moment it was grabbed.
        Slot _drag;
        OVRInput.Controller _dragController;
        float _dragMin, _dragMax;

        const float Height = 0.03f;
        const float Gap = 0.005f;
        const float Pad = 0.008f;               // card around the rows
        const float TextHeight = 0.0115f;
        const float Radius = 0.006f;
        const float RayLength = 2;

        static readonly Color Card = new(0.06f, 0.07f, 0.09f, 0.9f);
        static readonly Color Normal = new(0.17f, 0.18f, 0.22f, 0.95f);
        static readonly Color Accent = new(0.16f, 0.45f, 0.9f, 1);
        static readonly Color Hover = new(0.27f, 0.29f, 0.35f, 1);
        static readonly Color HoverBorder = new(0.45f, 0.72f, 1, 1);
        static readonly Color Track = new(0.12f, 0.13f, 0.16f, 0.95f);
        static readonly Color SliderFill = new(0.16f, 0.42f, 0.82f, 0.9f);
        static readonly Color Off = new(0.12f, 0.12f, 0.14f, 0.7f);
        static readonly Color Pressed = new(0.2f, 0.7f, 0.35f, 1);
        static readonly Color Refused = new(0.8f, 0.2f, 0.2f, 1);
        static readonly Color TextOff = new(0.45f, 0.47f, 0.52f);
        const float FlashSeconds = 0.35f;
        const float FeedbackSeconds = 6;

        Slot _flash;
        bool _flashOk;
        float _flashUntil;
        string _feedback = "";
        float _feedbackAt = -100;

        /// <summary>What the last press did ("Pressed: Solve"), for a few seconds; empty after.</summary>
        public string Feedback => Time.unscaledTime - _feedbackAt < FeedbackSeconds ? _feedback : "";

        Transform _root;
        float _width;
        readonly List<Slot> _slots = new();
        readonly Dictionary<Collider, Slot> _byCollider = new();
        UiRect _card;
        Transform _trackingSpace;
        LineRenderer _leftRay, _rightRay;
        Material _rayMaterial;

        /// <summary>Creates the bar centred at <paramref name="localPosition"/>, <paramref name="width"/> wide.</summary>
        /// <summary>A controller ray points at a button.</summary>
        public bool Hovered { get; private set; }

        public void Init(Transform parent, Vector3 localPosition, float width)
        {
            _root = new GameObject("Buttons").transform;
            _root.SetParent(parent, false);
            _root.localPosition = localPosition;
            _width = width;
            _card = UiRect.Create(_root, "Menu card", Vector3.zero, Vector2.one * 0.01f, Card, 0.014f);
            _card.SetBorder(new Color(1, 1, 1, 0.08f), 0.0008f);
            _card.Order = -1;
            _rayMaterial = new Material(Resources.Load<Shader>("P2ProRgb")) { mainTexture = Texture2D.whiteTexture };
            _leftRay = CreateRay("Left pointer");
            _rightRay = CreateRay("Right pointer");
        }

        public void Show(params Button[][] rows)
        {
            int total = 0;
            foreach (var row in rows) total += row.Length;
            while (_slots.Count < total) _slots.Add(CreateSlot());
            int next = 0;
            for (int r = 0; r < rows.Length; r++)
            {
                var row = rows[r];
                float w = (_width - Gap * (row.Length - 1)) / Mathf.Max(1, row.Length);
                for (int i = 0; i < row.Length; i++)
                {
                    var slot = _slots[next++];
                    slot.Root.SetActive(true);
                    slot.Button = row[i];
                    slot.Width = w;
                    slot.Root.transform.localPosition = new Vector3(-_width / 2 + w / 2 + i * (w + Gap), (rows.Length - 1 - r) * (Height + Gap), 0);
                    slot.Background.SetSize(new Vector2(w, Height));
                    slot.Fill.gameObject.SetActive(row[i].IsSlider);
                    slot.Handle.gameObject.SetActive(row[i].IsSlider);
                    if (row[i].IsSlider) UpdateSlider(slot);
                    else slot.Text.Text = row[i].Label;
                }
            }
            for (int i = next; i < _slots.Count; i++) _slots[i].Root.SetActive(false);
            float height = rows.Length * Height + (rows.Length - 1) * Gap + 2 * Pad;
            _card.SetSize(new Vector2(_width + 2 * Pad, height));
            _card.transform.localPosition = new Vector3(0, height / 2 - Height / 2 - Pad, 0.001f);
            if (_drag != null && (!_drag.Root.activeSelf || !_drag.Button.IsSlider)) _drag = null;
        }

        // Fill up to the value, the handle on it, and the label.
        void UpdateSlider(Slot slot)
        {
            var b = slot.Button;
            float value = b.Get();
            float f = Mathf.InverseLerp(b.Min, b.Max, value);
            if (slot == _drag) f = Mathf.InverseLerp(_dragMin, _dragMax, value);
            float w = slot.Width;
            float fill = Mathf.Max(w * f, 2 * Radius);      // keeps the rounded end
            slot.Fill.transform.localPosition = new Vector3(-w / 2 + fill / 2, 0, -0.0005f);
            slot.Fill.SetSize(new Vector2(fill, Height));
            slot.Handle.transform.localPosition = new Vector3(Mathf.Clamp(-w / 2 + w * f, -w / 2 + 0.004f, w / 2 - 0.004f), 0, -0.0007f);
            slot.Text.Text = b.Format(value);
        }

        Slot CreateSlot()
        {
            var root = new GameObject("Button");
            root.transform.SetParent(_root, false);
            var slot = new Slot
            {
                Root = root,
                Background = UiRect.Create(root.transform, "Background", Vector3.zero, new Vector2(0.1f, Height), Normal, Radius, collider: true),
                Text = PanelText.Create(root.transform, new Vector3(0, 0, -0.001f), "Label", TextAnchor.MiddleCenter, TextHeight),
                Fill = UiRect.Create(root.transform, "Fill", Vector3.zero, new Vector2(0.01f, Height), SliderFill, Radius),
                Handle = UiRect.Create(root.transform, "Handle", Vector3.zero, new Vector2(0.003f, Height * 0.6f), Color.white, 0.0015f),
            };
            slot.Fill.Order = 1;
            slot.Handle.Order = 2;
            slot.Fill.gameObject.SetActive(false);
            slot.Handle.gameObject.SetActive(false);
            _byCollider[slot.Background.Collider] = slot;
            return slot;
        }

        LineRenderer CreateRay(string name)
        {
            var line = new GameObject(name).AddComponent<LineRenderer>();
            line.sharedMaterial = _rayMaterial;
            line.widthMultiplier = 0.0015f;
            line.positionCount = 2;
            line.useWorldSpace = true;
            return line;
        }

        void Update()
        {
            if (_trackingSpace == null)
            {
                var rig = FindAnyObjectByType<OVRCameraRig>();
                if (rig != null) _trackingSpace = rig.trackingSpace;
            }
            bool visible = _root.gameObject.activeInHierarchy;
            Slot hovered = null;
            hovered = Point(OVRInput.Controller.LTouch, _leftRay, visible) ?? hovered;
            hovered = Point(OVRInput.Controller.RTouch, _rightRay, visible) ?? hovered;
            Hovered = hovered != null || _drag != null;

            foreach (var slot in _slots)
            {
                bool enabled = slot.Button.Enabled || slot.Button.IsSlider;
                bool hover = slot == hovered || slot == _drag;
                var colour = slot == _flash && Time.unscaledTime < _flashUntil ? (_flashOk ? Pressed : Refused)
                    : slot.Button.IsSlider ? Track
                    : hover ? (slot.Button.Highlight ? Color.Lerp(Accent, Color.white, 0.15f) : Hover)
                    : !enabled ? Off
                    : slot.Button.Highlight ? Accent : Normal;
                slot.Background.Colour = colour;
                slot.Background.SetBorder(hover ? HoverBorder : Color.clear, 0.0012f);
                slot.Text.Colour = enabled ? Color.white : TextOff;
                if (slot.Root.activeSelf && slot.Button.IsSlider) UpdateSlider(slot);
            }
        }

        // Ray from one controller; clicks the button it points at when the trigger is pulled.
        Slot Point(OVRInput.Controller controller, LineRenderer ray, bool visible)
        {
            bool connected = visible && OVRInput.IsControllerConnected(controller);
            ray.enabled = connected;
            if (!connected) return null;

            var position = OVRInput.GetLocalControllerPosition(controller);
            var rotation = OVRInput.GetLocalControllerRotation(controller);
            if (_trackingSpace != null)
            {
                position = _trackingSpace.TransformPoint(position);
                rotation = _trackingSpace.rotation * rotation;
            }
            var direction = rotation * Vector3.forward;
            if (_drag != null && _dragController == controller)
            {
                if (OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger, controller)) Drag(new Ray(position, direction));
                else _drag = null;
            }
            Slot slot = null;
            float length = RayLength;
            if (Physics.Raycast(position, direction, out var hit, RayLength) && _byCollider.TryGetValue(hit.collider, out slot))
                length = hit.distance;
            ray.SetPosition(0, position);
            ray.SetPosition(1, position + direction * length);

            if (slot != null && OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, controller))
            {
                if (slot.Button.IsSlider)
                {
                    _drag = slot;
                    _dragController = controller;
                    _dragMin = slot.Button.Min;
                    _dragMax = slot.Button.Max;
                    Drag(new Ray(position, direction));
                }
                else
                {
                    _flash = slot;
                    _flashUntil = Time.unscaledTime + FlashSeconds;
                    _flashOk = slot.Button.Enabled;
                    _feedbackAt = Time.unscaledTime;
                    if (slot.Button.Enabled)
                    {
                        _feedback = "Pressed: " + slot.Button.Label;
                        Debug.Log("[P2Pro] button " + slot.Button.Label);
                        slot.Button.OnClick?.Invoke();
                    }
                    else
                    {
                        _feedback = $"\"{slot.Button.Label}\" is not available now" + (string.IsNullOrEmpty(slot.Button.Disabled) ? "." : ": " + slot.Button.Disabled);
                        Debug.Log("[P2Pro] button refused: " + slot.Button.Label);
                    }
                }
            }
            return slot;
        }

        // The held slider takes the value where the ray crosses it (clamped to its ends).
        void Drag(Ray ray)
        {
            var background = _drag.Background.transform;   // its scale is its size: x runs -0.5 .. 0.5
            if (!new Plane(background.forward, background.position).Raycast(ray, out float d)) return;
            float x = background.InverseTransformPoint(ray.GetPoint(d)).x;
            _drag.Button.Set(Mathf.Lerp(_dragMin, _dragMax, Mathf.Clamp01(x + 0.5f)));
        }

        void OnDestroy()
        {
            if (_leftRay != null) Destroy(_leftRay.gameObject);
            if (_rightRay != null) Destroy(_rightRay.gameObject);
            Destroy(_rayMaterial);
        }
    }
}
