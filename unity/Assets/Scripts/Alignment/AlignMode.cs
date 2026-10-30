using System.Text;
using UnityEngine;
using UnityEngine.Rendering;

namespace P2Pro
{
    /// <summary>
    /// Calibration and viewing, driven by the pointer buttons above the debug panel.
    /// <list type="bullet">
    /// <item>View: the thermal image on the room mesh, either live (<see cref="ThermalLive"/>,
    /// nothing kept) or painted into the room's thermal map (<see cref="ThermalMap"/>, kept when
    /// out of view), or off.</item>
    /// <item>Calibrate: the user looks at a warm, bright rectangle (a screen with a white page)
    /// and presses Capture. Both images freeze; if the target is found in both, the overlay on
    /// the RGB image is fitted to its corners. The user fine-tunes it with the sticks and presses
    /// Save point, which stores it with the distance measured from the room depth. Points near,
    /// mid and far give <see cref="ThermalAlignment"/>.</item>
    /// <item>Points wizard (from Calibrate): on a frozen frame the user marks the same features
    /// on the thermal and the RGB image (<see cref="PointsWizard"/>); over a few frames at
    /// different distances these fit the thermal camera's lens model (<see cref="ThermalLens"/>),
    /// which then replaces the mapping for everything projected on the room.</item>
    /// </list>
    /// Sticks while frozen: left = move, right up / down = size, right left / right = turn,
    /// hold a grip = fine steps. Y while viewing = next room mode (live, paint, flat); X = thermal
    /// surfaces / thermal wireframe; B = HUD (<see cref="Hud"/>).
    /// </summary>
    public class AlignMode : MonoBehaviour
    {
        enum State { View, Calibrate, Frozen, Points }
        // Values are stored in PlayerPrefs: append new modes, do not reorder.
        enum RoomMode { Live, Paint, Off, Flat }
        static readonly string[] RoomModeNames = { "Room: live", "Room: paint + keep", "Room: off", "Flat: no 3D" };
        static readonly RoomMode[] RoomModeOrder = { RoomMode.Live, RoomMode.Paint, RoomMode.Flat, RoomMode.Off };
        static readonly RoomMode[] QuickRoomModes = { RoomMode.Live, RoomMode.Paint, RoomMode.Flat };   // Y
        const string RoomModeKey = "p2pro.roomMode";

        const float ProbeInterval = 0.2f;
        const float MoveSpeed = 0.15f;          // RGB widths per second at full stick
        const float ScaleSpeed = 0.4f;          // ln(scale) per second
        const float RollSpeed = 30;             // degrees per second
        const float FineFactor = 0.15f;
        const float DeadZone = 0.15f;
        const float MaxAutoRmsPx = 25;          // corner fit error above which the auto fit is ignored
        const float FallbackDistance = 1.5f;
        const float ThermalHalfFovDeg = 28;
        const float ConfirmSeconds = 3;
        static readonly Vector2[] CentreSpread =
            { new(0, 0), new(-0.03f, 0), new(0.03f, 0), new(0, -0.03f), new(0, 0.03f) };

        ThermalPanel _panel;
        PanelText _text;
        ButtonBar _buttons;
        DepthProbe _probe;
        ThermalMap _map;
        ThermalLive _live;
        ThermalFlat _flat;
        RoomMode _roomMode;
        readonly OverlayStyle _style = new();
        LatencyEstimator _latency;
        readonly TargetFinder _finder = new();
        ThermalAlignment _alignment;
        PointsWizard _wizard;

        State _state;
        float _distance = -1;
        float _nextProbe;
        float _nextText;
        float _nextRate;
        float _confirmResetUntil;
        float _confirmLensUntil;
        int _thermalFps, _rgbFps, _thermalFrames, _rgbFrames;
        string _note = "";

        // Frozen frame being aligned.
        Pose _frozenPose;
        float _frozenDistance;
        Vector2 _targetViewport;                // RGB viewport point to measure the distance at
        ThermalAlignment.Mapping _edit;
        string _auto = "";
        bool _autoFitted;
        bool _touched;

        public void Init(ThermalPanel panel, PanelText text, ButtonBar buttons)
        {
            _panel = panel;
            _text = text;
            _buttons = buttons;
            _probe = gameObject.AddComponent<DepthProbe>();
            _alignment = ThermalAlignment.Load();
            _latency = gameObject.AddComponent<LatencyEstimator>();
            _latency.Init(panel, _alignment);
            _map = gameObject.AddComponent<ThermalMap>();
            _map.Init(panel, _alignment, _latency, _style);
            _live = gameObject.AddComponent<ThermalLive>();
            _live.Init(panel, _alignment, _latency, _style);
            _flat = gameObject.AddComponent<ThermalFlat>();
            _flat.Init(panel, _alignment, _latency, _style, () => _distance > 0 ? _distance : FallbackDistance);
            gameObject.AddComponent<PointerTemperature>().Init(_map, buttons);
            _wizard = gameObject.AddComponent<PointsWizard>();
            _wizard.Init(panel, _alignment, buttons, _probe);
            panel.SceneRange = () => _map.SceneRange;
            gameObject.AddComponent<Hud>().Init(panel, HudText);
            SetRoomMode((RoomMode)PlayerPrefs.GetInt(RoomModeKey, (int)RoomMode.Paint));
            Debug.Log($"[P2Pro] alignment: {_alignment.Points.Count} points from {ThermalAlignment.FilePath}; lens " +
                      (_alignment.Lens.Solved ? _alignment.Lens.ToString() : $"not solved ({_alignment.Lens.Pairs.Count} pairs)"));
        }

        void Update()
        {
            if (_panel == null || _panel.Thermal == null) return;
            if (_state == State.Frozen) Edit();
            if (_state == State.View && OVRInput.GetDown(OVRInput.Button.Four)) NextRoomMode(QuickRoomModes);
            if (OVRInput.GetDown(OVRInput.Button.Three)) _style.ToggleWire();
            Probe();
            UpdateOverlay();
            UpdateRates();
            if (Time.unscaledTime >= _nextText)
            {
                _nextText = Time.unscaledTime + 0.25f;
                _text.Text = Status();
                ShowButtons();
            }
        }

        void ShowButtons()
        {
            switch (_state)
            {
                case State.View:
                    string latency = _latency.Auto
                        ? $"Latency {_latency.LatencyMs:F0} ms auto"
                        : $"Latency {_latency.LatencyMs:F0} ms manual";
                    _buttons.Show(new[]
                    {
                        new ButtonBar.Button("Calibrate", () => SetState(State.Calibrate), !_alignment.Ready),
                        new ButtonBar.Button(RoomModeNames[(int)_roomMode], () => NextRoomMode(RoomModeOrder)),
                        new ButtonBar.Button("Clear map", _map.Clear),
                        new ButtonBar.Button("Palette: " + _panel.PaletteName, _panel.NextPalette),
                        new ButtonBar.Button("Range: " + _panel.RangeName, _panel.NextRange),
                    }, new[]
                    {
                        new ButtonBar.Button(latency, ToggleLatencyMode),
                        new ButtonBar.Button("-10 ms", () => NudgeLatency(-LatencyEstimator.StepMs)),
                        new ButtonBar.Button("+10 ms", () => NudgeLatency(LatencyEstimator.StepMs)),
                        new ButtonBar.Button("Recenter panel", _panel.PlaceInFrontOfHead),
                        new ButtonBar.Button(_panel.Sharpening.Label, ToggleSharpening),
                        new ButtonBar.Button(_panel.FlatField.Label, ToggleFlatField),
                    }, new[]
                    {
                        new ButtonBar.Button(_style.BlendLabel, _style.NextBlend),
                        new ButtonBar.Button(_style.FilterLabel, () => _style.NextFilter(_panel.RangeMin, _panel.RangeMax)),
                        new ButtonBar.Button(_style.WireLabel, _style.ToggleWire),
                    }, new[]
                    {
                        ButtonBar.Button.Slider(() => _style.Threshold, _style.SetThreshold,
                            _panel.RangeMin, _panel.RangeMax, _ => _style.ThresholdLabel),
                        ButtonBar.Button.Slider(() => _style.Opacity, _style.SetOpacity,
                            OverlayStyle.MinOpacity, 1, v => $"Opacity {v * 100:F0}%"),
                    }, RangeSliders());
                    break;
                case State.Calibrate:
                    bool confirming = Time.unscaledTime < _confirmResetUntil;
                    bool confirmingLens = Time.unscaledTime < _confirmLensUntil;
                    _buttons.Show(new[]
                    {
                        new ButtonBar.Button("Capture", Freeze, _alignment.Points.Count == 0),
                        new ButtonBar.Button("Undo last point", RemoveLast),
                        new ButtonBar.Button(confirming ? "Sure? Click again" : "Reset all points", ResetAll),
                        new ButtonBar.Button("Done", () => SetState(State.View)),
                    }, new[]
                    {
                        new ButtonBar.Button($"Points wizard ({_alignment.Lens.Pairs.Count})", StartPoints, _alignment.Points.Count > 0 && !_alignment.Lens.Solved),
                        new ButtonBar.Button("Solve lens", _wizard.Solve),
                        new ButtonBar.Button(confirmingLens ? "Sure? Click again" : "Reset lens", ResetLens),
                    });
                    break;
                case State.Points:
                    _buttons.Show(new[]
                    {
                        new ButtonBar.Button("Next frame", () => SetState(State.Calibrate), _wizard.FramePairs >= 4),
                        new ButtonBar.Button("Undo pair", _wizard.Undo),
                        new ButtonBar.Button($"Turn thermal ({_wizard.Roll:F0} deg)", _wizard.Turn),
                        new ButtonBar.Button($"Solve lens ({_alignment.Lens.Pairs.Count})", _wizard.Solve,
                            _alignment.Lens.Pairs.Count >= ThermalLens.MinPairs),
                        new ButtonBar.Button("Done", () => SetState(State.View)),
                    });
                    break;
                case State.Frozen:
                    _buttons.Show(new[]
                    {
                        new ButtonBar.Button("Save point", Save, true),
                        new ButtonBar.Button("Cancel", Cancel),
                    });
                    break;
            }
        }

        // Min and max of the palette (switch the range to Manual). Their span covers what the
        // camera and the room show, with a margin, so a slider has room to move both ways.
        ButtonBar.Button[] RangeSliders()
        {
            var t = _panel.Thermal;
            float lo = Mathf.Min(_panel.RangeMin, t.Min), hi = Mathf.Max(_panel.RangeMax, t.Max);
            if (_map.SceneRange is { } scene) { lo = Mathf.Min(lo, scene.x); hi = Mathf.Max(hi, scene.y); }
            lo = Mathf.Floor(lo - 5);
            hi = Mathf.Ceil(hi + 5);
            return new[]
            {
                ButtonBar.Button.Slider(() => _panel.RangeMin, v => _panel.SetManualRange(Mathf.Min(v, _panel.RangeMax - 0.5f), _panel.RangeMax),
                    lo, hi, v => $"Palette min {v:F1} C"),
                ButtonBar.Button.Slider(() => _panel.RangeMax, v => _panel.SetManualRange(_panel.RangeMin, Mathf.Max(v, _panel.RangeMin + 0.5f)),
                    lo, hi, v => $"Palette max {v:F1} C"),
            };
        }

        void SetState(State state)
        {
            if (_state is State.Frozen or State.Points && state != _state) _panel.Unfreeze();
            if (_state == State.Points && state != State.Points) _wizard.End();
            _state = state;
            _note = "";
            _nextText = 0;
            Debug.Log("[P2Pro] state " + state);
        }

        void SetRoomMode(RoomMode mode)
        {
            _roomMode = mode;
            _live.Visible = mode == RoomMode.Live;
            // The map keeps baking unless off: the pointer reads its temperatures in every mode.
            _map.Baking = mode != RoomMode.Off;
            _map.Visible = mode == RoomMode.Paint;
            _flat.Visible = mode == RoomMode.Flat;
            PlayerPrefs.SetInt(RoomModeKey, (int)mode);
            _nextText = 0;
            Debug.Log("[P2Pro] " + RoomModeNames[(int)mode]);
        }

        // From a mode missing in the order (Off for Y) the next is the first.
        void NextRoomMode(RoomMode[] order)
        {
            int i = System.Array.IndexOf(order, _roomMode);
            SetRoomMode(order[(i + 1) % order.Length]);
        }

        // Off -> on starts learning from scratch.
        void ToggleFlatField()
        {
            _panel.FlatField.Enabled = !_panel.FlatField.Enabled;
            _nextText = 0;
        }

        void ToggleSharpening()
        {
            _panel.Sharpening.Enabled = !_panel.Sharpening.Enabled;
            _nextText = 0;
        }

        // Auto -> manual keeps the current value, so the nudge buttons start from it.
        void ToggleLatencyMode()
        {
            if (_latency.Auto) _latency.ManualMs = Mathf.Round(_latency.LatencyMs);
            else _latency.ResetEstimate();
            _latency.Auto = !_latency.Auto;
            _nextText = 0;
        }

        void NudgeLatency(float ms)
        {
            if (_latency.Auto) _latency.ManualMs = Mathf.Round(_latency.LatencyMs);
            _latency.Auto = false;
            _latency.ManualMs = Mathf.Clamp(_latency.ManualMs + ms, 0, 300);
            _nextText = 0;
            Debug.Log($"[P2Pro] latency {_latency.ManualMs} ms (manual)");
        }

        void Freeze()
        {
            var source = _panel.Rgb.Source;
            if (source == null || !source.IsPlaying || !_panel.Freeze())
            {
                _note = "No thermal or RGB image yet.";
                return;
            }
            SetState(State.Frozen);
            _frozenPose = WorldCameraPose();
            _frozenDistance = -1;
            _targetViewport = new Vector2(0.5f, 0.5f);
            _edit = _alignment.TryEvaluate(_distance > 0 ? _distance : FallbackDistance, out var m) && ThermalAlignment.Plausible(m)
                ? m : DefaultMapping();
            _touched = _autoFitted = false;

            if (!_finder.FindThermal(_panel.Thermal.Celsius, out var thermalQuad, out var reason))
            {
                _auto = "Target not found in thermal (" + reason + "): align by hand.";
                return;
            }
            _auto = "Looking for the target in RGB...";
            // The camera texture still holds the frozen image this frame.
            AsyncGPUReadback.Request(source.GetTexture(), 0, TextureFormat.RGBA32, request =>
            {
                if (request.hasError || _state != State.Frozen) return;
                var size = new Vector2Int(request.width, request.height);
                if (!_finder.FindRgb(request.GetData<Color32>(), size.x, size.y, out var rgbQuad, out var rgbReason))
                {
                    _auto = "Target not found in RGB (" + rgbReason + "): align by hand.";
                    return;
                }
                var fit = ThermalAlignment.FitCorners(thermalQuad, rgbQuad, size, out float rms);
                if (rms > MaxAutoRmsPx || !ThermalAlignment.Plausible(fit))
                {
                    _auto = $"Thermal and RGB targets do not match ({rms:F0} px): align by hand.";
                    return;
                }
                _auto = $"Fitted automatically to the target corners ({rms:F1} px). Check, adjust if needed.";
                _autoFitted = true;
                _targetViewport = new Vector2(rgbQuad.Center.x / size.x, rgbQuad.Center.y / size.y);
                _frozenDistance = -1;
                if (!_touched) _edit = fit;
                Debug.Log($"[P2Pro] auto fit {rms:F1} px: {fit}");
            });
        }

        void StartPoints()
        {
            var source = _panel.Rgb.Source;
            if (source == null || !source.IsPlaying || !_panel.Freeze())
            {
                _note = "No thermal or RGB image yet.";
                return;
            }
            SetState(State.Points);
            _wizard.Begin(WorldCameraPose());
        }

        void ResetLens()
        {
            if (Time.unscaledTime >= _confirmLensUntil)
            {
                _confirmLensUntil = Time.unscaledTime + ConfirmSeconds;
                _nextText = 0;
                return;
            }
            _confirmLensUntil = 0;
            _alignment.Lens = new ThermalLens();
            _alignment.Save();
            _note = "Lens model and its point pairs deleted.";
            _nextText = 0;
            Debug.Log("[P2Pro] lens reset");
        }

        void Save()
        {
            if (_frozenDistance <= 0)
            {
                _note = "No distance yet: wait a second (the room depth must cover the target).";
                return;
            }
            if (!ThermalAlignment.Plausible(_edit))
            {
                _note = "The overlay is off the RGB image: move it back onto the target.";
                return;
            }
            string source = _autoFitted ? (_touched ? "auto+manual" : "auto") : "manual";
            _alignment.Add(_frozenDistance, _edit, source);
            _alignment.Save();
            Debug.Log($"[P2Pro] point saved at {_frozenDistance:F2} m ({source}): {_edit}");
            SetState(State.Calibrate);
            _note = $"Saved point at {_frozenDistance:F2} m.";
        }

        void Cancel()
        {
            SetState(State.Calibrate);
            _note = "Frame discarded.";
        }

        void RemoveLast()
        {
            if (_alignment.Points.Count == 0) return;
            var removed = _alignment.Points[^1];
            _alignment.Points.RemoveAt(_alignment.Points.Count - 1);
            _alignment.Save();
            _note = $"Deleted the point at {removed.DistanceM:F2} m.";
            Debug.Log("[P2Pro] " + _note);
        }

        void ResetAll()
        {
            if (Time.unscaledTime >= _confirmResetUntil)
            {
                _confirmResetUntil = Time.unscaledTime + ConfirmSeconds;
                _nextText = 0;
                return;
            }
            _confirmResetUntil = 0;
            _alignment.Points.Clear();
            _alignment.Save();
            _note = "All points deleted.";
            _nextText = 0;
            Debug.Log("[P2Pro] alignment reset");
        }

        void Edit()
        {
            var move = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick);
            var size = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick);
            if (move.magnitude < DeadZone) move = Vector2.zero;
            if (size.magnitude < DeadZone) size = Vector2.zero;
            if (move == Vector2.zero && size == Vector2.zero) return;

            _touched = true;
            bool fine = OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger) > 0.5f ||
                        OVRInput.Get(OVRInput.Axis1D.SecondaryHandTrigger) > 0.5f;
            float k = (fine ? FineFactor : 1) * Time.deltaTime;
            _edit.U = Mathf.Clamp(_edit.U + move.x * MoveSpeed * k, -0.2f, 1.2f);
            _edit.V = Mathf.Clamp(_edit.V + move.y * MoveSpeed * k / ThermalAlignment.RgbAspect, -0.2f, 1.2f);
            _edit.Scale = Mathf.Clamp(_edit.Scale * Mathf.Exp(size.y * ScaleSpeed * k), 0.15f, 1.9f);
            _edit.RollDeg -= size.x * RollSpeed * k;
        }

        // Distance to what the RGB camera looks at: live at the image centre, or for a frozen
        // frame at the target (or the centre) seen from the frozen pose.
        void Probe()
        {
            var source = _panel.Rgb.Source;
            if (source == null || !source.IsPlaying || _state == State.Points) return;  // the wizard uses the probe
            if (_state == State.Frozen)
            {
                if (_frozenDistance > 0) return;
                // Busy until the previous answer is in; an answer for an outdated point is dropped.
                var target = _targetViewport;
                _probe.Request(Rays(source, target, _frozenPose), d =>
                {
                    if (_state == State.Frozen && _targetViewport == target) _frozenDistance = DepthProbe.Median(d);
                });
                return;
            }
            if (Time.unscaledTime < _nextProbe) return;
            if (_probe.Request(Rays(source, new Vector2(0.5f, 0.5f), WorldCameraPose()), d => _distance = DepthProbe.Median(d)))
                _nextProbe = Time.unscaledTime + ProbeInterval;
        }

        static Ray[] Rays(Meta.XR.PassthroughCameraAccess source, Vector2 viewport, Pose pose)
        {
            var rays = new Ray[CentreSpread.Length];
            for (int i = 0; i < rays.Length; i++) rays[i] = source.ViewportPointToRay(viewport + CentreSpread[i], pose);
            return rays;
        }

        // PassthroughCameraAccess already returns the pose in world space (tracking space
        // transform included), unlike OVRPlugin's head pose in RgbProjection.
        Pose WorldCameraPose() => _panel.Rgb.Source.GetCameraPose();

        void UpdateOverlay()
        {
            if (_state == State.Points)
                _panel.ShowOverlay(null);       // both images as captured, to mark on
            else if (_state == State.Frozen)
                _panel.ShowOverlay(_edit);
            else if (_alignment.TryEvaluate(_distance > 0 ? _distance : FallbackDistance, out var m) && ThermalAlignment.Plausible(m))
                _panel.ShowOverlay(m);
            else
                _panel.ShowOverlay(_state == State.Calibrate ? DefaultMapping() : null);
        }

        // Centred, sized from the nominal fields of view.
        ThermalAlignment.Mapping DefaultMapping()
        {
            float scale = 0.7f;
            var source = _panel.Rgb.Source;
            if (source != null && source.IsPlaying)
            {
                var edge = source.WorldToViewportPoint(
                    new Vector3(Mathf.Tan(ThermalHalfFovDeg * Mathf.Deg2Rad), 0, 1), new Pose(Vector3.zero, Quaternion.identity));
                if (edge.x > 0.5f) scale = 2 * (edge.x - 0.5f);
            }
            return new ThermalAlignment.Mapping { U = 0.5f, V = 0.5f, Scale = scale };
        }

        void UpdateRates()
        {
            if (Time.unscaledTime < _nextRate) return;
            _nextRate = Time.unscaledTime + 1;
            _thermalFps = _panel.Thermal.FrameCount - _thermalFrames;
            _rgbFps = _panel.Rgb.FrameCount - _rgbFrames;
            _thermalFrames = _panel.Thermal.FrameCount;
            _rgbFrames = _panel.Rgb.FrameCount;
        }

        // A few lines for the HUD: modes, the camera's readings, how the room is painted.
        string HudText()
        {
            var t = _panel.Thermal;
            var sb = new StringBuilder();
            string room = _roomMode switch
            {
                RoomMode.Live => "LIVE", RoomMode.Paint => "PAINT", RoomMode.Flat => "FLAT", _ => "OFF",
            };
            sb.AppendLine($"{room}  {(_style.Wire ? "wire" : "surface")}  {_panel.PaletteName}  {_panel.RangeName}");
            if (_state is State.Frozen or State.Points) sb.AppendLine("Thermal: frozen");
            else if (t.FrameCount == 0 || _thermalFps == 0) sb.AppendLine("Thermal: " + t.Status);
            else sb.AppendLine($"Thermal {_thermalFps} fps   min {t.Min:F1}  max {t.Max:F1}  centre {t.Center:F1} C");
            sb.AppendLine($"Distance {(_distance > 0 ? $"{_distance:F2} m" : "-")}   RGB {_rgbFps} fps");
            if (_roomMode == RoomMode.Paint)
                sb.AppendLine($"Painted {_map.Baked} frames" + (_map.Skipped.Length > 0 ? $"  ({_map.Skipped})" : ""));
            sb.AppendLine(_alignment.Lens.Solved ? $"Lens {_alignment.Lens.RmsPx:F1} px" : $"Mapping {_alignment.Points.Count} points");
            sb.Append("Y mode  X wire  sticks palette / range  B hide");
            return sb.ToString();
        }

        string Status()
        {
            var t = _panel.Thermal;
            var sb = new StringBuilder();
            string thermal = _state is State.Frozen or State.Points ? "frozen"
                : t.FrameCount == 0 || _thermalFps == 0 ? t.Status
                : $"{_thermalFps} fps, {t.Min:F1}..{t.Max:F1} C";
            string depth = _distance > 0 ? $"{_distance:F2} m" : "no depth yet";
            sb.AppendLine($"Thermal: {thermal}    RGB: {_rgbFps} fps    Distance to centre of view: {depth}");
            sb.AppendLine("Calibration: " + Points() + "    Menu = hide / show panel    Y = room mode    X = wire    B = HUD    Point a controller anywhere = its temperature");

            switch (_state)
            {
                case State.View when !_alignment.Ready:
                    sb.AppendLine("Not calibrated yet. Point at Calibrate and pull the trigger.");
                    break;
                case State.View:
                    string scene = _map.SceneRange is { } r ? $"{r.x:F1}..{r.y:F1} C" : "nothing yet";
                    sb.AppendLine(_roomMode switch
                    {
                        RoomMode.Live => "Room: live. Only what the thermal camera sees now is coloured.",
                        RoomMode.Paint => $"Room: paint + keep. {_map.Baked} frames painted" +
                                          (_map.Skipped.Length > 0 ? $" (waiting: {_map.Skipped})" : "") +
                                          $"; what leaves the view keeps its last temperature. Scene range {scene}.",
                        RoomMode.Flat => $"Flat: the thermal picture at {(_distance > 0 ? _distance : FallbackDistance):F1} m, not projected on the room.",
                        _ => "Room: off.",
                    });
                    sb.AppendLine(_latency.Auto
                        ? (_latency.Estimated.HasValue
                            ? $"Latency {_latency.LatencyMs:F0} ms, measured automatically ({_latency.Pairs} head movements)."
                            : $"Latency {_latency.LatencyMs:F0} ms default. To measure it: shake your head left-right for ~10 s ({_latency.Pairs}/40).")
                        : $"Latency {_latency.LatencyMs:F0} ms, set by hand. If the colours trail when you turn, add; if they lead, subtract.");
                    break;
                case State.Calibrate:
                    sb.AppendLine("1. Look at a warm, bright rectangle (a screen with a white page) so it is in the middle of the view.");
                    sb.AppendLine("2. Hold still and press Capture.   Repeat near (~0.5 m), mid (~1.2 m) and far (~2.5 m), then Done.");
                    sb.AppendLine("Edges off? Points wizard: mark the same features on both images, fits the thermal lens (needs 6+ pairs; 15-30 over 3-4 frames is better).");
                    break;
                case State.Points:
                    sb.AppendLine($"Points wizard: {_wizard.FramePairs} pairs on this frame, {_alignment.Lens.Pairs.Count} in all.");
                    sb.AppendLine("Point at a sharp feature on the LEFT (thermal) image, pull the trigger, then at the same spot on the RIGHT (RGB) image.");
                    sb.AppendLine("Good: corners of a warm screen, lamp, radiator, window frame, hand. Spread pairs over the whole image, corners and edges too.");
                    sb.AppendLine("Then Next frame at another distance (near ~0.6 m, far ~2.5 m), and Solve lens. Undo pair removes the last one.");
                    sb.AppendLine("Thermal image turned wrong? Turn thermal rotates it 90 deg (display only, remembered).");
                    if (_wizard.Note.Length > 0) sb.AppendLine(_wizard.Note);
                    break;
                case State.Frozen:
                    string at = _frozenDistance > 0 ? $"{_frozenDistance:F2} m" : "measuring distance...";
                    sb.AppendLine($"Frozen at {at}.  {_auto}");
                    sb.AppendLine("Make the coloured overlay on the right image match the RGB picture:");
                    sb.AppendLine("left stick = move, right stick up/down = size, left/right = turn, hold grip = fine.  Then Save point.");
                    break;
            }
            if (_note.Length > 0) sb.AppendLine(_note);
            return sb.ToString();
        }

        string Points()
        {
            string lens = _alignment.Lens.Solved ? $", lens {_alignment.Lens.Used} pairs {_alignment.Lens.RmsPx:F1} px" : "";
            if (_alignment.Points.Count == 0) return (_alignment.Lens.Solved ? "mapping none" : "none") + lens;
            var sb = new StringBuilder();
            sb.Append(_alignment.Points.Count).Append(_alignment.Points.Count == 1 ? " point at" : " points at");
            foreach (var p in _alignment.Points) sb.Append($" {p.DistanceM:F2}");
            return sb.Append(" m").Append(lens).ToString();
        }
    }
}
