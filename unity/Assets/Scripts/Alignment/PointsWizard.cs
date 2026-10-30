using System.Collections.Generic;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Point pairs for the thermal lens model (<see cref="ThermalLens"/>), marked on a frozen
    /// frame: the user points a controller at a feature on the thermal image (left) and pulls the
    /// trigger, then at the same feature on the RGB image (right), in either order. The RGB point
    /// is measured in the room depth from the frozen camera pose, giving its position in the RGB
    /// camera's frame; with the thermal pixel that is one pair. Pairs from several frames (other
    /// views, other distances) add up; Solve fits the lens to all of them.
    /// A dot follows the controller on the images; each pair gets a numbered colour on both.
    /// The thermal image is shown turned like the RGB one (the mapping's roll), plus quarter turns
    /// the user adds (remembered); display only, pairs keep the camera's own pixels.
    /// </summary>
    public class PointsWizard : MonoBehaviour
    {
        const float MarkerSize = 0.004f;
        const float CursorSize = 0.003f;
        const float AutoRollDistance = 1.5f;
        const string TurnKey = "p2pro.pointsTurn";
        static readonly Color[] Colours =
        {
            new(1, 0.3f, 0.3f), new(0.3f, 1, 0.3f), new(0.3f, 0.6f, 1), new(1, 1, 0.3f),
            new(1, 0.3f, 1), new(0.3f, 1, 1), new(1, 0.6f, 0.2f), new(0.7f, 0.5f, 1),
        };

        ThermalPanel _panel;
        ThermalAlignment _alignment;
        ButtonBar _buttons;
        DepthProbe _probe;
        Transform _trackingSpace;
        Pose _pose;                             // RGB camera at the frozen frame (world)
        bool _active;

        Vector2? _thermal, _rgb;                // marked, waiting for the other half
        bool _measuring;
        Transform _pendingThermal, _pendingRgb;
        readonly Transform[] _cursors = new Transform[2];
        int _turns;                             // quarter turns the user added
        float _autoRoll;
        Vector2 _pendingThermalUv, _pendingRgbUv;
        readonly List<(int pair, Transform thermal, Transform rgb, Vector2 thermalUv, Vector2 rgbUv)> _markers = new();
        Material _cursorMaterial;
        readonly Dictionary<int, Material> _colourMaterials = new();

        /// <summary>Pairs marked on the current frame.</summary>
        public int FramePairs { get; private set; }
        public string Note { get; private set; } = "";

        /// <summary>Display roll of the thermal image, degrees.</summary>
        public float Roll => Mathf.Repeat(_autoRoll + 90 * _turns + 180, 360) - 180;

        public void Init(ThermalPanel panel, ThermalAlignment alignment, ButtonBar buttons, DepthProbe probe)
        {
            _panel = panel;
            _alignment = alignment;
            _buttons = buttons;
            _probe = probe;
            _turns = PlayerPrefs.GetInt(TurnKey, 0);
            _cursorMaterial = ColourMaterial(-1);
            for (int i = 0; i < 2; i++) _cursors[i] = Sphere("Points cursor", _cursorMaterial, CursorSize);
        }

        /// <summary>Starts marking on the frame the panel just froze, seen from <paramref name="rgbPose"/>.</summary>
        public void Begin(Pose rgbPose)
        {
            _pose = rgbPose;
            _active = true;
            FramePairs = 0;
            Note = "";
            ClearPending();
            _autoRoll = _alignment.TryEvaluate(AutoRollDistance, out var m) ? m.RollDeg : 0;
            _panel.ThermalRoll = Roll;
        }

        /// <summary>Turns the thermal image a quarter turn further (display only).</summary>
        public void Turn()
        {
            _turns = (_turns + 1) % 4;
            PlayerPrefs.SetInt(TurnKey, _turns);
            PlayerPrefs.Save();
            if (_active) _panel.ThermalRoll = Roll;
        }

        public void End()
        {
            _active = false;
            _panel.ThermalRoll = null;
            ClearPending();
            foreach (var m in _markers) { Destroy(m.thermal.gameObject); Destroy(m.rgb.gameObject); }
            _markers.Clear();
            foreach (var c in _cursors) c.gameObject.SetActive(false);
        }

        public void Undo()
        {
            if (_thermal.HasValue || _rgb.HasValue) { ClearPending(); return; }
            var pairs = _alignment.Lens.Pairs;
            if (pairs.Count == 0) return;
            int last = pairs.Count - 1;
            pairs.RemoveAt(last);
            _alignment.Save();
            int k = _markers.FindIndex(m => m.pair == last);
            if (k >= 0)
            {
                Destroy(_markers[k].thermal.gameObject);
                Destroy(_markers[k].rgb.gameObject);
                _markers.RemoveAt(k);
                FramePairs--;
            }
            Note = $"Deleted pair {last + 1}.";
        }

        public void Solve()
        {
            string log = _alignment.Lens.Solve();
            _alignment.Save();
            Note = _alignment.Lens.Solved ? "Lens: " + _alignment.Lens + ". " + log : log;
            Debug.Log("[P2Pro] lens solve: " + log + (_alignment.Lens.Solved ? " -> " + _alignment.Lens : ""));
        }

        void Update()
        {
            if (!_active) return;
            // Markers sit on the images as they are turned now.
            foreach (var m in _markers)
            {
                m.thermal.position = _panel.ImagePoint(true, m.thermalUv);
                m.rgb.position = _panel.ImagePoint(false, m.rgbUv);
            }
            if (_pendingThermal != null) _pendingThermal.position = _panel.ImagePoint(true, _pendingThermalUv);
            if (_pendingRgb != null) _pendingRgb.position = _panel.ImagePoint(false, _pendingRgbUv);
            int c = 0;
            foreach (var controller in new[] { OVRInput.Controller.RTouch, OVRInput.Controller.LTouch })
            {
                var cursor = _cursors[c++];
                cursor.gameObject.SetActive(false);
                if (!OVRInput.IsControllerConnected(controller) || !TryGetRay(controller, out var ray)) continue;
                if (!_panel.HitImage(ray, out bool thermal, out var uv)) continue;
                cursor.gameObject.SetActive(true);
                cursor.position = _panel.ImagePoint(thermal, uv);
                if (!_buttons.Hovered && !_measuring && OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, controller))
                    Mark(thermal, uv);
            }
        }

        void Mark(bool thermal, Vector2 uv)
        {
            Note = "";
            ref var slot = ref thermal ? ref _pendingThermal : ref _pendingRgb;
            if (slot == null) slot = Sphere("Points pending", _cursorMaterial, MarkerSize);
            slot.position = _panel.ImagePoint(thermal, uv);
            if (thermal) _thermal = _pendingThermalUv = uv; else _rgb = _pendingRgbUv = uv;
            if (_thermal.HasValue && _rgb.HasValue) Measure();
        }

        // Depth at the RGB point from the frozen pose, then the pair.
        void Measure()
        {
            var source = _panel.Rgb.Source;
            if (source == null) return;
            var ray = source.ViewportPointToRay(_rgb.Value, _pose);
            var thermalUv = _thermal.Value;
            var rgbUv = _rgb.Value;
            _measuring = _probe.Request(new[] { ray }, d =>
            {
                _measuring = false;
                if (!_active) return;
                if (d[0] <= 0)
                {
                    Note = "No room depth at that RGB point: mark features on surfaces the scan sees (not windows, mirrors, sky).";
                    ClearPending();
                    return;
                }
                var world = ray.GetPoint(d[0]);
                var rgb = Quaternion.Inverse(_pose.rotation) * (world - _pose.position);
                var pairs = _alignment.Lens.Pairs;
                pairs.Add(new ThermalLens.Pair
                {
                    Rgb = rgb,
                    ThermalPx = new Vector2(thermalUv.x * ThermalCamera.Width, thermalUv.y * ThermalCamera.Height),
                });
                _alignment.Save();
                int index = pairs.Count - 1;
                var material = ColourMaterial(index);
                _markers.Add((index,
                    Sphere($"Pair {index + 1} thermal", material, MarkerSize, _panel.ImagePoint(true, thermalUv)),
                    Sphere($"Pair {index + 1} RGB", material, MarkerSize, _panel.ImagePoint(false, rgbUv)),
                    thermalUv, rgbUv));
                FramePairs++;
                ClearPending();
                Note = $"Pair {index + 1} at {d[0]:F2} m.";
                Debug.Log($"[P2Pro] lens pair {index + 1}: thermal {thermalUv:F3}, rgb {rgbUv:F3}, {d[0]:F2} m");
            });
            if (!_measuring) Note = "Depth busy, mark the RGB point again.";
        }

        void ClearPending()
        {
            _thermal = _rgb = null;
            if (_pendingThermal != null) Destroy(_pendingThermal.gameObject);
            if (_pendingRgb != null) Destroy(_pendingRgb.gameObject);
            _pendingThermal = _pendingRgb = null;
        }

        bool TryGetRay(OVRInput.Controller controller, out Ray ray)
        {
            ray = default;
            if (_trackingSpace == null)
            {
                var rig = FindAnyObjectByType<OVRCameraRig>();
                if (rig == null) return false;
                _trackingSpace = rig.trackingSpace;
            }
            var position = _trackingSpace.TransformPoint(OVRInput.GetLocalControllerPosition(controller));
            var rotation = _trackingSpace.rotation * OVRInput.GetLocalControllerRotation(controller);
            ray = new Ray(position, rotation * Vector3.forward);
            return true;
        }

        Transform Sphere(string name, Material material, float size, Vector3? position = null)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = name;
            Destroy(go.GetComponent<Collider>());
            go.GetComponent<MeshRenderer>().sharedMaterial = material;
            go.transform.SetParent(_panel.Root, true);
            go.transform.localScale = Vector3.one * size;
            if (position.HasValue) go.transform.position = position.Value;
            return go.transform;
        }

        // White for the cursor and pending marks (-1), else the pair's colour.
        Material ColourMaterial(int pair)
        {
            int key = pair < 0 ? -1 : pair % Colours.Length;
            if (_colourMaterials.TryGetValue(key, out var m)) return m;
            m = new Material(Resources.Load<Shader>("P2ProText")) { mainTexture = Texture2D.whiteTexture };
            m.color = key < 0 ? Color.white : Colours[key];
            m.renderQueue = 4000;
            _colourMaterials[key] = m;
            return m;
        }

        void OnDestroy()
        {
            End();
            foreach (var c in _cursors) if (c != null) Destroy(c.gameObject);
            foreach (var m in _colourMaterials.Values) Destroy(m);
        }
    }
}
