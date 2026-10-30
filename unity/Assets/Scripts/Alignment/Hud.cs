using System;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Heads-up display in the lower left of the view: a few lines of what the app is doing
    /// (from <see cref="AlignMode"/>) and the palette bar with the colour range in °C. Follows the
    /// head with a short lag so it does not shake. Shown / hidden with B (<see cref="ControllerGuide"/>;
    /// remembered between launches).
    /// </summary>
    public class Hud : MonoBehaviour
    {
        const string Key = "p2pro.hud";
        static readonly Vector3 Offset = new(-0.17f, -0.1f, 0.55f);    // from the eyes, metres
        const float LineHeight = 0.011f;
        const float BarWidth = 0.12f, BarHeight = 0.008f;
        const float FollowRate = 8;             // 1/s
        const float TextInterval = 0.25f;

        ThermalPanel _panel;
        Func<string> _status;
        Transform _root, _head;
        PanelText _text, _min, _max;
        Material _barMaterial;
        bool _visible;
        bool _placed;
        float _nextText;

        public void Init(ThermalPanel panel, Func<string> status)
        {
            _panel = panel;
            _status = status;
            _root = new GameObject("HUD").transform;
            _text = PanelText.Create(_root, Vector3.zero, "HUD text", TextAnchor.UpperLeft, LineHeight);
            _text.AlwaysOnTop();

            var bar = GameObject.CreatePrimitive(PrimitiveType.Quad);
            bar.name = "HUD palette";
            Destroy(bar.GetComponent<Collider>());
            _barMaterial = new Material(Resources.Load<Shader>("P2ProText"));
            _barMaterial.SetFloat("_TexColour", 1);
            _barMaterial.SetFloat("_ZTest", (float)UnityEngine.Rendering.CompareFunction.Always);
            _barMaterial.renderQueue = 4000;
            bar.GetComponent<MeshRenderer>().sharedMaterial = _barMaterial;
            bar.transform.SetParent(_root, false);
            bar.transform.localScale = new Vector3(BarWidth, BarHeight, 1);
            bar.transform.localPosition = new Vector3(BarWidth / 2, BarHeight / 2 + 0.004f, 0);
            _min = PanelText.Create(_root, new Vector3(0, BarHeight + 0.006f + LineHeight, 0), "HUD min", TextAnchor.UpperLeft, LineHeight);
            _max = PanelText.Create(_root, new Vector3(BarWidth, BarHeight + 0.006f + LineHeight, 0), "HUD max", TextAnchor.UpperRight, LineHeight);
            _min.AlwaysOnTop();
            _max.AlwaysOnTop();

            Visible = PlayerPrefs.GetInt(Key, 1) == 1;
        }

        public bool Visible
        {
            get => _visible;
            set
            {
                _visible = value;
                _root.gameObject.SetActive(value);
                PlayerPrefs.SetInt(Key, value ? 1 : 0);
                PlayerPrefs.Save();
                _nextText = 0;
            }
        }

        void Update()
        {
            if (!_visible || !TryGetHead(out var head)) return;

            var target = new Pose(head.position + head.rotation * Offset, head.rotation);
            float k = _placed ? 1 - Mathf.Exp(-FollowRate * Time.unscaledDeltaTime) : 1;
            _placed = true;
            _root.SetPositionAndRotation(Vector3.Lerp(_root.position, target.position, k),
                                         Quaternion.Slerp(_root.rotation, target.rotation, k));

            if (Time.unscaledTime < _nextText) return;
            _nextText = Time.unscaledTime + TextInterval;
            _text.Text = _status();
            _barMaterial.mainTexture = _panel.Palette;
            _min.Text = $"{_panel.RangeMin:F1} C";
            _max.Text = $"{_panel.RangeMax:F1} C";
        }

        bool TryGetHead(out Transform head)
        {
            if (_head == null)
            {
                var rig = FindAnyObjectByType<OVRCameraRig>();
                if (rig != null) _head = rig.centerEyeAnchor;
            }
            head = _head;
            return head != null;
        }

        void OnDestroy()
        {
            if (_root != null) Destroy(_root.gameObject);
            Destroy(_barMaterial);
        }
    }
}
