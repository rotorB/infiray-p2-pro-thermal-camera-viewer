using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Train mode: calibrates the thermal camera against the headset. The user puts a warm,
    /// bright rectangle in view (a monitor or laptop screen with a white page), keeps it still
    /// and walks around it, near and far, pausing now and then. The status text under the debug
    /// view shows what is still missing; the solver runs in the background once there is enough
    /// data and saves the result when it is good.
    /// Controls: menu button (left) = start / stop, A = solve now, left stick click = reset.
    /// </summary>
    public class TrainMode : MonoBehaviour
    {
        const float AutoSolveSeconds = 5;
        const float AcceptRmsPx = 1.0f;
        const int MinRgbViews = 6;
        const float MinSpreadDeg = 15;
        const float MaxTriangulationMm = 10;
        const int MinThermal = 40;
        const int MinPerDistance = 5;
        const int MinImageCells = 6;

        public ThermalCalibration Current { get; private set; }
        /// <summary>Raised when a new calibration is saved.</summary>
        public event System.Action<ThermalCalibration> Calibrated;

        CalibrationCollector _collector;
        PanelText _text;
        readonly Transform[] _markers = new Transform[4];
        Material _markerMaterial;
        Texture2D _markerTexture;

        bool _active;
        Task<CalibrationSolver.Result> _solve;
        CalibrationSolver.Result _last;
        string _lastNote = "";
        float _nextAutoSolve;
        float _nextText;

        public void Init(ThermalCamera thermal, HeadsetRgbCamera rgb, Transform panelRoot, Vector3 textPosition)
        {
            _collector = gameObject.AddComponent<CalibrationCollector>();
            _collector.Init(thermal, rgb);

            Current = ThermalCalibration.Load();
            Debug.Log(Current != null ? "[P2Pro] calibration loaded: " + Current : "[P2Pro] no calibration yet");

            _text = PanelText.Create(panelRoot, textPosition, "Train status");

            _markerTexture = new Texture2D(1, 1);
            _markerTexture.SetPixel(0, 0, new Color(0.1f, 1f, 0.3f));
            _markerTexture.Apply();
            _markerMaterial = new Material(Resources.Load<Shader>("P2ProRgb")) { mainTexture = _markerTexture };
            for (int k = 0; k < 4; k++)
            {
                var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere.name = "Target corner " + k;
                Destroy(sphere.GetComponent<Collider>());
                sphere.GetComponent<MeshRenderer>().sharedMaterial = _markerMaterial;
                sphere.transform.localScale = Vector3.one * 0.015f;
                sphere.SetActive(false);
                _markers[k] = sphere.transform;
            }
        }

        void Update()
        {
            if (_collector == null) return;

            if (OVRInput.GetDown(OVRInput.Button.Start)) SetActive(!_active);
            if (_active && OVRInput.GetDown(OVRInput.Button.PrimaryThumbstick)) Reset();
            if (_active && OVRInput.GetDown(OVRInput.Button.One)) StartSolve();
            if (_active && Ready(out _) && Time.unscaledTime >= _nextAutoSolve) StartSolve();

            if (_solve != null && _solve.IsCompleted) FinishSolve();

            for (int k = 0; k < 4; k++)
            {
                bool show = _active && _collector.HasCorners3D;
                if (_markers[k].gameObject.activeSelf != show) _markers[k].gameObject.SetActive(show);
                if (show) _markers[k].position = _collector.Corners3D[k];
            }

            if (Time.unscaledTime >= _nextText)
            {
                _nextText = Time.unscaledTime + 0.25f;
                _text.Text = Status();
            }
        }

        void SetActive(bool active)
        {
            _active = active;
            _collector.Collecting = active;
            if (active) Reset();
            Debug.Log("[P2Pro] train mode " + (active ? "on" : "off"));
        }

        void Reset()
        {
            _collector.Clear();
            _last = null;
            _lastNote = "";
            _nextAutoSolve = 0;
        }

        bool Ready(out string missing)
        {
            var c = _collector;
            var parts = new List<string>();
            if (c.RgbViews < MinRgbViews) parts.Add($"RGB views {c.RgbViews}/{MinRgbViews} (pause to let it capture)");
            else if (c.ViewSpreadDeg < MinSpreadDeg) parts.Add($"walk sideways: view spread {c.ViewSpreadDeg:F0}/{MinSpreadDeg} deg");
            if (c.HasCorners3D && c.TriangulationRmsMm > MaxTriangulationMm) parts.Add("corners disagree: target moved? reset");
            if (c.Thermal.Count < MinThermal) parts.Add($"thermal frames {c.Thermal.Count}/{MinThermal}");
            if (c.DistanceBins[0] < MinPerDistance) parts.Add("come closer");
            if (c.DistanceBins[2] < MinPerDistance) parts.Add("step back");
            int cells = 0;
            foreach (int b in c.ImageBins) if (b > 0) cells++;
            if (cells < MinImageCells) parts.Add($"move the target around the thermal image ({cells}/{MinImageCells} areas)");
            missing = string.Join("; ", parts);
            return parts.Count == 0;
        }

        void StartSolve()
        {
            _nextAutoSolve = Time.unscaledTime + AutoSolveSeconds;
            if (_solve != null || !_collector.HasCorners3D || _collector.Thermal.Count < 10) return;
            var input = new CalibrationSolver.Input
            {
                Corners3D = (Vector3[])_collector.Corners3D.Clone(),
                Observations = new List<CalibrationSolver.Observation>(_collector.Thermal),
                LatencyStepMs = CalibrationCollector.LatencyStepMs,
                RgbViews = _collector.RgbViews,
            };
            _lastNote = "solving...";
            _solve = Task.Run(() => CalibrationSolver.Solve(input));
        }

        void FinishSolve()
        {
            var task = _solve;
            _solve = null;
            if (task.IsFaulted)
            {
                _lastNote = "solve failed: " + task.Exception?.GetBaseException().Message;
                Debug.LogError("[P2Pro] " + _lastNote);
                return;
            }
            _last = task.Result;
            Debug.Log($"[P2Pro] calibration solve:\n{_last.Log}{_last.Calibration}");
            bool good = _last.Calibration.RmsPx < AcceptRmsPx && _last.Used >= 30 && Ready(out _);
            if (good)
            {
                _last.Calibration.Save();
                Current = _last.Calibration;
                _lastNote = "SAVED";
                Debug.Log("[P2Pro] calibration saved to " + ThermalCalibration.FilePath);
                Calibrated?.Invoke(Current);
            }
            else
            {
                _lastNote = _last.Calibration.RmsPx < AcceptRmsPx ? "not saved: collect the missing data" : "not saved: error too high";
            }
        }

        string Status()
        {
            if (!_active)
            {
                return Current != null
                    ? $"Calibrated {Current.CreatedUtc?.Substring(0, 10)}: RMS {Current.RmsPx:F2} px, latency {Current.LatencyMs:F0} ms.  Menu = Train again"
                    : "Not calibrated.  Menu (left controller) = Train";
            }
            var c = _collector;
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("TRAIN   Menu = stop, A = solve now, left stick click = reset");
            sb.AppendLine("Target: warm, bright rectangle (screen with a white page). Keep it still, walk around it.");
            sb.AppendLine($"RGB: {c.RgbState}.  views {c.RgbViews}, spread {c.ViewSpreadDeg:F0} deg, " +
                          $"corners +-{c.TriangulationRmsMm:F1} mm, size {c.TargetSize()}");
            sb.AppendLine($"Thermal: {c.ThermalState}.  frames {c.Thermal.Count}: near {c.DistanceBins[0]}, " +
                          $"mid {c.DistanceBins[1]}, far {c.DistanceBins[2]}");
            sb.AppendLine(Ready(out var missing) ? "Data: complete" : "Missing: " + missing);
            if (_last != null) sb.AppendLine($"Result ({_last.Used}/{_last.Total} frames): {_last.Calibration}");
            if (_lastNote.Length > 0) sb.AppendLine(_lastNote);
            return sb.ToString();
        }

        void OnDestroy()
        {
            foreach (var m in _markers) if (m != null) Destroy(m.gameObject);
            Destroy(_markerMaterial);
            Destroy(_markerTexture);
        }
    }
}
