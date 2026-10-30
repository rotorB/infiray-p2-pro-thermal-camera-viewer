using System;
using System.Collections.Generic;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Rows of buttons on the debug panel, pressed by pointing a controller at them and pulling
    /// its index trigger. Each controller draws a ray; the button under it lights up.
    /// The owner calls <see cref="Show"/> with the current buttons whenever they may change;
    /// rows stack downwards from the bar's position.
    /// </summary>
    public class ButtonBar : MonoBehaviour
    {
        public struct Button
        {
            public string Label;
            public Action OnClick;
            public bool Highlight;              // drawn in the accent colour (the main next step)

            public Button(string label, Action onClick, bool highlight = false)
            {
                Label = label;
                OnClick = onClick;
                Highlight = highlight;
            }
        }

        class Slot
        {
            public GameObject Root;
            public MeshRenderer Background;
            public PanelText Text;
            public Button Button;
        }

        const float Height = 0.045f;
        const float Gap = 0.01f;
        const float RayLength = 2;

        static readonly Color Normal = new(0.18f, 0.18f, 0.22f);
        static readonly Color Accent = new(0.1f, 0.45f, 0.85f);
        static readonly Color Hover = new(0.95f, 0.6f, 0.1f);

        Transform _root;
        float _width;
        readonly List<Slot> _slots = new();
        readonly Dictionary<Collider, Slot> _byCollider = new();
        readonly Dictionary<Color, Material> _materials = new();
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
            _rayMaterial = ColourMaterial(Color.white);
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
                    slot.Root.transform.localPosition = new Vector3(-_width / 2 + w / 2 + i * (w + Gap), -r * (Height + Gap), 0);
                    slot.Background.transform.localScale = new Vector3(w, Height, 1);
                    slot.Text.Text = row[i].Label;
                }
            }
            for (int i = next; i < _slots.Count; i++) _slots[i].Root.SetActive(false);
        }

        Slot CreateSlot()
        {
            var root = new GameObject("Button");
            root.transform.SetParent(_root, false);
            var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);   // keeps its MeshCollider for the pointer
            quad.transform.SetParent(root.transform, false);
            var slot = new Slot
            {
                Root = root,
                Background = quad.GetComponent<MeshRenderer>(),
                Text = PanelText.Create(root.transform, new Vector3(0, 0, -0.001f), "Label", TextAnchor.MiddleCenter, 0.016f),
            };
            _byCollider[quad.GetComponent<Collider>()] = slot;
            return slot;
        }

        LineRenderer CreateRay(string name)
        {
            var line = new GameObject(name).AddComponent<LineRenderer>();
            line.sharedMaterial = _rayMaterial;
            line.widthMultiplier = 0.002f;
            line.positionCount = 2;
            line.useWorldSpace = true;
            return line;
        }

        Material ColourMaterial(Color colour)
        {
            if (_materials.TryGetValue(colour, out var material)) return material;
            var texture = new Texture2D(1, 1);
            texture.SetPixel(0, 0, colour);
            texture.Apply();
            material = new Material(Resources.Load<Shader>("P2ProRgb")) { mainTexture = texture };
            _materials[colour] = material;
            return material;
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
            Hovered = hovered != null;

            foreach (var slot in _slots)
            {
                var colour = slot == hovered ? Hover : slot.Button.Highlight ? Accent : Normal;
                slot.Background.sharedMaterial = ColourMaterial(colour);
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
            Slot slot = null;
            float length = RayLength;
            if (Physics.Raycast(position, direction, out var hit, RayLength) && _byCollider.TryGetValue(hit.collider, out slot))
                length = hit.distance;
            ray.SetPosition(0, position);
            ray.SetPosition(1, position + direction * length);

            if (slot != null && OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, controller))
            {
                Debug.Log("[P2Pro] button " + slot.Button.Label);
                slot.Button.OnClick?.Invoke();
            }
            return slot;
        }

        void OnDestroy()
        {
            if (_leftRay != null) Destroy(_leftRay.gameObject);
            if (_rightRay != null) Destroy(_rightRay.gameObject);
            foreach (var m in _materials.Values)
            {
                Destroy(m.mainTexture);
                Destroy(m);
            }
        }
    }
}
