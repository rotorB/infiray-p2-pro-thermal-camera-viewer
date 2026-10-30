using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Thermal flashlight: while on, the thermal overlays (live, painted room, flat picture) show
    /// only inside a cone from the right controller, like a torch in the dark; the rest of the
    /// room stays plain passthrough. Off, everything is as usual. A on the right controller
    /// switches it (<see cref="ControllerGuide"/>); every launch starts off. The cone's width is remembered.
    /// Sets globals read by P2ProOverlay.hlsl.
    /// </summary>
    public class ThermalFlashlight : MonoBehaviour
    {
        public const float MinAngle = 3, MaxAngle = 45;    // half-angle, degrees
        const float Rim = 0.25f;                            // soft edge, fraction of the half-angle
        const string AngleKey = "p2pro.flashlightAngle";

        static readonly int OnId = Shader.PropertyToID("_FlashOn");
        static readonly int PosId = Shader.PropertyToID("_FlashPos");
        static readonly int DirId = Shader.PropertyToID("_FlashDir");
        static readonly int CosId = Shader.PropertyToID("_FlashCos");

        Transform _trackingSpace;
        float _angle;

        public bool On { get; private set; }

        /// <summary>Half-angle of the cone, degrees.</summary>
        public float Angle
        {
            get => _angle;
            set
            {
                _angle = Mathf.Clamp(value, MinAngle, MaxAngle);
                PlayerPrefs.SetFloat(AngleKey, _angle);
            }
        }

        public string Label => On ? $"Flashlight: on ({_angle:F0} deg)" : "Flashlight: off";

        void Awake()
        {
            _angle = Mathf.Clamp(PlayerPrefs.GetFloat(AngleKey, 15), MinAngle, MaxAngle);
            Shader.SetGlobalFloat(OnId, 0);
        }

        public void Toggle()
        {
            On = !On;
            Shader.SetGlobalFloat(OnId, On ? 1 : 0);
            Debug.Log("[P2Pro] flashlight " + (On ? "on" : "off"));
        }

        void Update()
        {
            if (!On) return;
            if (_trackingSpace == null)
            {
                var rig = FindAnyObjectByType<OVRCameraRig>();
                if (rig == null) return;
                _trackingSpace = rig.trackingSpace;
            }
            var c = OVRInput.Controller.RTouch;
            var position = _trackingSpace.TransformPoint(OVRInput.GetLocalControllerPosition(c));
            var direction = _trackingSpace.rotation * OVRInput.GetLocalControllerRotation(c) * Vector3.forward;
            Shader.SetGlobalVector(PosId, position);
            Shader.SetGlobalVector(DirId, direction);
            Shader.SetGlobalVector(CosId, new Vector4(
                Mathf.Cos(_angle * Mathf.Deg2Rad), Mathf.Cos(_angle * (1 - Rim) * Mathf.Deg2Rad)));
        }

        void OnDestroy()
        {
            Shader.SetGlobalFloat(OnId, 0);
            PlayerPrefs.Save();
        }
    }
}
