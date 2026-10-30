using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Every controller button in one table (<see cref="Bind"/>): what it does, what it is set to
    /// now and what it can be. From the same table:
    /// <list type="bullet">
    /// <item>a press runs the action and shows a toast in the middle of the view for a moment:
    /// the button, what it switches, all its choices with the new one in brackets; a button
    /// that cannot act now says why instead;</item>
    /// <item>looking at a controller shows a legend above it: each of its buttons with its
    /// current setting.</item>
    /// </list>
    /// </summary>
    public class ControllerGuide : MonoBehaviour
    {
        const float ToastSeconds = 2.2f;
        static readonly Vector3 ToastOffset = new(0, -0.07f, 0.6f);     // from the eyes, metres
        const float LegendLookDeg = 28;         // shown while the controller is this close to the gaze
        const float LegendMaxDistance = 0.75f;
        const float LegendAbove = 0.07f;
        const float LineHeight = 0.009f;
        const string KeyColour = "#ffc833";

        class Binding
        {
            public OVRInput.Button Button;
            public bool Left;
            public string Key, Name;
            public Func<string> Value;
            public Action Press;
            public Func<string[]> Options;
            public Func<string> Unavailable;
        }

        readonly List<Binding> _bindings = new();
        Transform _head, _leftHand, _rightHand;
        PanelText _toast, _leftLegend, _rightLegend;
        float _toastUntil;

        void Awake()
        {
            _toast = PanelText.Create(transform, Vector3.zero, "Controller toast", TextAnchor.MiddleCenter, 0.016f);
            _leftLegend = PanelText.Create(transform, Vector3.zero, "Left controller legend", TextAnchor.LowerCenter, LineHeight);
            _rightLegend = PanelText.Create(transform, Vector3.zero, "Right controller legend", TextAnchor.LowerCenter, LineHeight);
            foreach (var t in new[] { _toast, _leftLegend, _rightLegend })
            {
                t.AlwaysOnTop();
                t.gameObject.SetActive(false);
            }
        }

        /// <summary>
        /// One button. <paramref name="press"/> null = information only (the trigger), not polled.
        /// <paramref name="unavailable"/> returns why the button cannot act now, or null.
        /// </summary>
        public void Bind(OVRInput.Button button, bool left, string key, string name, Func<string> value,
            Action press, Func<string[]> options = null, Func<string> unavailable = null)
        {
            _bindings.Add(new Binding
            {
                Button = button, Left = left, Key = key, Name = name, Value = value,
                Press = press, Options = options, Unavailable = unavailable,
            });
        }

        void Update()
        {
            foreach (var b in _bindings)
            {
                if (b.Press == null || !OVRInput.GetDown(b.Button)) continue;
                string why = b.Unavailable?.Invoke();
                if (why != null) Toast($"{Key(b.Key)}  {b.Name.ToUpperInvariant()}\nnot now: {why}");
                else
                {
                    b.Press();
                    Toast($"{Key(b.Key)}  {b.Name.ToUpperInvariant()}\n{Choices(b)}");
                }
            }
            if (!TryGetAnchors()) return;

            if (_toast.gameObject.activeSelf)
            {
                if (Time.unscaledTime > _toastUntil) _toast.gameObject.SetActive(false);
                else _toast.transform.SetPositionAndRotation(_head.position + _head.rotation * ToastOffset, _head.rotation);
            }
            Legend(_leftLegend, _leftHand, true);
            Legend(_rightLegend, _rightHand, false);
        }

        void Toast(string text)
        {
            _toast.Text = text;
            _toast.gameObject.SetActive(true);
            _toastUntil = Time.unscaledTime + ToastSeconds;
            Debug.Log("[P2Pro] " + text.Replace('\n', ' '));
        }

        // All choices, the current one in brackets; or just the value.
        static string Choices(Binding b)
        {
            string value = b.Value();
            var options = b.Options?.Invoke();
            if (options == null || options.Length == 0) return value;
            var sb = new StringBuilder();
            foreach (string o in options)
            {
                if (sb.Length > 0) sb.Append("   ");
                sb.Append(value.StartsWith(o, StringComparison.OrdinalIgnoreCase) ? $"[ {o.ToUpperInvariant()} ]" : o);
            }
            return sb.ToString();
        }

        // Above the controller, facing the eyes, while it is looked at.
        void Legend(PanelText legend, Transform hand, bool left)
        {
            var controller = left ? OVRInput.Controller.LTouch : OVRInput.Controller.RTouch;
            var to = hand.position - _head.position;
            bool show = OVRInput.IsControllerConnected(controller) && to.magnitude < LegendMaxDistance &&
                        Vector3.Angle(_head.forward, to) < LegendLookDeg;
            legend.gameObject.SetActive(show);
            if (!show) return;
            var position = hand.position + Vector3.up * LegendAbove;
            legend.transform.SetPositionAndRotation(position, Quaternion.LookRotation(position - _head.position, Vector3.up));
            var sb = new StringBuilder();
            foreach (var b in _bindings)
            {
                if (b.Left != left) continue;
                if (sb.Length > 0) sb.Append('\n');
                sb.Append($"{Key(b.Key)}  {b.Name}: {b.Value()}");
            }
            legend.Text = sb.ToString();
        }

        static string Key(string key) => $"<color={KeyColour}>{key}</color>";

        bool TryGetAnchors()
        {
            if (_head == null)
            {
                var rig = FindAnyObjectByType<OVRCameraRig>();
                if (rig == null) return false;
                _head = rig.centerEyeAnchor;
                _leftHand = rig.leftControllerAnchor;
                _rightHand = rig.rightControllerAnchor;
            }
            return _head != null && _leftHand != null && _rightHand != null;
        }
    }
}
