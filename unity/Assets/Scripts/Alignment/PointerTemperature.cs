using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Temperature where the controller points: a dot on the surface and its °C next to it.
    /// The surface is found by a depth ray march (<see cref="DepthProbe"/>); the temperature is
    /// read only from the room's thermal map (<see cref="ThermalMap.RequestTemperature"/>), the
    /// same °C the painted mesh shows. Both answers come back from the GPU a few frames late, so
    /// only the distance along the ray is taken from the probe: the dot sits on this frame's ray
    /// and follows the controller without lag. Steadied for reading: a missed depth or map answer
    /// keeps the last one for a moment, the value is smoothed over time and the text changes at
    /// most a few times a second. Right controller, or the left one when the right is off.
    /// </summary>
    public class PointerTemperature : MonoBehaviour
    {
        const float LabelSize = 0.02f;          // metres per metre of distance (about 1.1°)
        const float MarkerSize = 0.006f;
        const float HoldSeconds = 0.5f;         // keep the last surface / value through misses
        const float SmoothSeconds = 0.25f;      // time constant of the value
        const float SnapCelsius = 3;            // a bigger change (new object) skips smoothing
        const float JumpMetres = 0.15f;         // a bigger move of the surface point skips smoothing
        const float TextInterval = 0.2f;

        ThermalMap _map;
        ButtonBar _buttons;
        DepthProbe _probe;
        Transform _trackingSpace;
        Transform _root;
        PanelText _label;

        float _distance;                        // along the ray, metres
        float _distanceTime = float.NegativeInfinity;
        float _celsius = float.NaN;             // smoothed
        float _celsiusTime = float.NegativeInfinity;
        float _nextText;

        public void Init(ThermalMap map, ButtonBar buttons)
        {
            _map = map;
            _buttons = buttons;
            _probe = gameObject.AddComponent<DepthProbe>();

            _root = new GameObject("Pointer temperature").transform;
            var marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            Destroy(marker.GetComponent<Collider>());
            marker.GetComponent<MeshRenderer>().sharedMaterial =
                new Material(Resources.Load<Shader>("P2ProRgb")) { mainTexture = Texture2D.whiteTexture };
            marker.transform.SetParent(_root, false);
            marker.transform.localScale = Vector3.one * MarkerSize / LabelSize;
            _label = PanelText.Create(_root, new Vector3(0.6f, 0.6f, 0), "Label", TextAnchor.LowerLeft, 1);
            _label.AlwaysOnTop();
            _root.gameObject.SetActive(false);
        }

        // LateUpdate: after OVRManager has read this frame's controller pose.
        void LateUpdate()
        {
            if (!TryGetRay(out var ray) || _buttons.Hovered)
            {
                _root.gameObject.SetActive(false);
                return;
            }

            _probe.Request(new[] { ray }, d =>
            {
                if (d[0] <= 0) return;
                // Small changes: smooth out depth noise; large ones: the pointer went elsewhere.
                _distance = Time.time - _distanceTime < HoldSeconds && Mathf.Abs(d[0] - _distance) < JumpMetres
                    ? Mathf.Lerp(_distance, d[0], 0.5f) : d[0];
                _distanceTime = Time.time;
            });
            if (Time.time - _distanceTime > HoldSeconds)
            {
                _root.gameObject.SetActive(false);
                return;
            }

            var surface = ray.GetPoint(_distance);
            _map.RequestTemperature(surface, OnTemperature);
            Show(surface);
        }

        void OnTemperature(float? celsius)
        {
            if (!celsius.HasValue) return;
            float now = Time.time;
            float dt = now - _celsiusTime;
            bool fresh = float.IsNaN(_celsius) || dt > HoldSeconds || Mathf.Abs(celsius.Value - _celsius) > SnapCelsius;
            _celsius = fresh ? celsius.Value : Mathf.Lerp(_celsius, celsius.Value, 1 - Mathf.Exp(-dt / SmoothSeconds));
            _celsiusTime = now;
        }

        bool TryGetRay(out Ray ray)
        {
            ray = default;
            var controller = OVRInput.IsControllerConnected(OVRInput.Controller.RTouch) ? OVRInput.Controller.RTouch
                : OVRInput.IsControllerConnected(OVRInput.Controller.LTouch) ? OVRInput.Controller.LTouch
                : OVRInput.Controller.None;
            if (controller == OVRInput.Controller.None) return false;
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

        void Show(Vector3 point)
        {
            var eye = Camera.main != null ? Camera.main.transform.position : point - Vector3.forward;
            var toEye = eye - point;
            float distance = toEye.magnitude;
            _root.gameObject.SetActive(true);
            // A little towards the eye so the dot is not hidden in the mesh.
            _root.position = point + toEye.normalized * Mathf.Min(0.02f, distance * 0.5f);
            _root.rotation = Quaternion.LookRotation(-toEye, Vector3.up);
            _root.localScale = Vector3.one * LabelSize * distance;

            if (Time.time < _nextText) return;
            _nextText = Time.time + TextInterval;
            bool recent = Time.time - _celsiusTime < HoldSeconds;
            _label.Text = recent ? $"{_celsius:F1} °C" : "no thermal data here yet";
        }

        void OnDestroy()
        {
            if (_root != null) Destroy(_root.gameObject);
        }
    }
}
