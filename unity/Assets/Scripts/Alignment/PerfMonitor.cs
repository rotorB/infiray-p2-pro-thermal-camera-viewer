using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.XR;
using Genesis.RoomScan;

namespace P2Pro
{
    /// <summary>
    /// Frame timing on the headset and the render settings that drive it, to see what costs.
    /// Every 2 s: frames per second against the display's rate, CPU main and render thread and
    /// GPU time per frame (FrameTimingManager; Player setting "Frame Timing Stats" on), logged
    /// with the app's mode and shown in the HUD. Settings switchable from the panel and
    /// remembered: foveated rendering (fewer pixels shaded towards the edge of each eye), eye
    /// resolution, MSAA, display rate, and how often the room scan integrates depth and rebuilds
    /// its mesh (both GPU-heavy; the room hardly changes once scanned).
    /// The GPU time itself is in the system log (VrApi "App=").
    /// </summary>
    public class PerfMonitor : MonoBehaviour
    {
        const float Interval = 2;
        const float DefaultRate = 90;
        const string Key = "p2pro.perf.";
        static readonly OVRManager.FoveatedRenderingLevel[] FoveationLevels =
        {
            OVRManager.FoveatedRenderingLevel.Off, OVRManager.FoveatedRenderingLevel.Low,
            OVRManager.FoveatedRenderingLevel.Medium, OVRManager.FoveatedRenderingLevel.High,
        };
        static readonly string[] FoveationNames = { "off", "low", "medium", "high" };
        // Room scan: depth integrations and mesh rebuilds per second; the last one pauses both.
        static readonly Vector2[] ScanRates = { new(3, 1), new(10, 3), new(30, 8), new(0.001f, 0.001f) };
        static readonly int[] PaintKeyframes = { 2, 1, 4 };
        static readonly int PaintKeyframesId = Shader.PropertyToID("_PaintKeyframes");
        static readonly float[] Resolutions = { 1, 0.9f, 0.8f, 0.7f };
        static readonly int[] MsaaSamples = { 4, 2, 1 };

        readonly FrameTiming[] _timing = new FrameTiming[1];
        Func<string> _context;
        int _foveation, _resolution, _msaa, _scan, _paint;
        int _frames, _timed;
        double _cpu, _render, _gpu, _worstMs;
        float _since;

        /// <summary>Last 2 s, one line; empty until the first.</summary>
        public string Summary { get; private set; } = "";
        public string FoveationLabel => "Foveation: " + FoveationNames[_foveation];
        public string ResolutionLabel => $"Resolution {Resolutions[_resolution] * 100:F0}%";
        public string MsaaLabel => MsaaSamples[_msaa] == 1 ? "MSAA off" : $"MSAA {MsaaSamples[_msaa]}x";
        public string RateLabel => $"Display {Rate:F0} Hz";
        public string PaintLabel => $"Paint detail: {PaintKeyframes[_paint]} frame{(PaintKeyframes[_paint] > 1 ? "s" : "")}";
        public string ScanLabel => _scan == ScanRates.Length - 1 ? "Scan: paused" : $"Scan {ScanRates[_scan].x:F0}/{ScanRates[_scan].y:F0} Hz";

        static float Rate => OVRManager.display != null ? OVRManager.display.displayFrequency : 0;

        public void Init(Func<string> context)
        {
            _context = context;
            _foveation = Mathf.Clamp(PlayerPrefs.GetInt(Key + "foveation", 3), 0, FoveationLevels.Length - 1);
            _resolution = Mathf.Clamp(PlayerPrefs.GetInt(Key + "resolution", 0), 0, Resolutions.Length - 1);
            _msaa = Mathf.Clamp(PlayerPrefs.GetInt(Key + "msaa", 0), 0, MsaaSamples.Length - 1);
            ApplyFoveation();
            ApplyResolution();
            ApplyMsaa();
            _scan = Mathf.Clamp(PlayerPrefs.GetInt(Key + "scan2", 0), 0, ScanRates.Length - 1);     // v1 started at 30/8 Hz
            _paint = Mathf.Clamp(PlayerPrefs.GetInt(Key + "paint2", 0), 0, PaintKeyframes.Length - 1);   // v1: test settings
            Shader.SetGlobalFloat(PaintKeyframesId, PaintKeyframes[_paint]);
            ApplyScan();
            // Every mode fits 90 Hz since the fixes above (GPU 4-8 ms); 72 / 120 selectable.
            float rate = PlayerPrefs.GetFloat(Key + "rate3", DefaultRate);
            var display = OVRManager.display;
            if (display != null && System.Array.Exists(display.displayFrequenciesAvailable ?? new float[0], r => Mathf.Abs(r - rate) < 1))
                display.displayFrequency = rate;
            Debug.Log("[P2Pro] " + RateLabel);
        }

        public void NextFoveation()
        {
            _foveation = (_foveation + 1) % FoveationLevels.Length;
            PlayerPrefs.SetInt(Key + "foveation", _foveation);
            ApplyFoveation();
        }

        public void NextResolution()
        {
            _resolution = (_resolution + 1) % Resolutions.Length;
            PlayerPrefs.SetInt(Key + "resolution", _resolution);
            ApplyResolution();
        }

        public void NextMsaa()
        {
            _msaa = (_msaa + 1) % MsaaSamples.Length;
            PlayerPrefs.SetInt(Key + "msaa", _msaa);
            ApplyMsaa();
        }

        public void NextScan()
        {
            _scan = (_scan + 1) % ScanRates.Length;
            PlayerPrefs.SetInt(Key + "scan2", _scan);
            ApplyScan();
        }

        /// <summary>Keyframes the painted room reprojects per pixel (P2ProThermalMesh): sharper with more, slower.</summary>
        public void NextPaint()
        {
            _paint = (_paint + 1) % PaintKeyframes.Length;
            PlayerPrefs.SetInt(Key + "paint2", _paint);
            Shader.SetGlobalFloat(PaintKeyframesId, PaintKeyframes[_paint]);
            Debug.Log("[P2Pro] " + PaintLabel);
        }

        // RoomScanner keeps its rates in private serialized fields.
        void ApplyScan()
        {
            var scanner = RoomScanner.Instance;
            if (scanner == null) return;
            const System.Reflection.BindingFlags Field = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            typeof(RoomScanner).GetField("integrationHz", Field)?.SetValue(scanner, ScanRates[_scan].x);
            typeof(RoomScanner).GetField("meshExtractionHz", Field)?.SetValue(scanner, ScanRates[_scan].y);
            Debug.Log("[P2Pro] " + ScanLabel);
        }

        /// <summary>Next display rate the headset offers.</summary>
        public void NextRate()
        {
            var display = OVRManager.display;
            var rates = display?.displayFrequenciesAvailable;
            if (rates == null || rates.Length == 0) return;
            var sorted = new List<float>(rates);
            sorted.Sort();
            int i = sorted.FindIndex(r => Mathf.Abs(r - display.displayFrequency) < 1);
            float next = sorted[(i + 1) % sorted.Count];
            display.displayFrequency = next;
            PlayerPrefs.SetFloat(Key + "rate3", next);
            Debug.Log($"[P2Pro] display rate {next} Hz");
        }

        // Meta's foveation feature (OpenXR XR_FB_foveation), as the project's OpenXR settings use it.
        void ApplyFoveation()
        {
            OVRManager.useDynamicFoveatedRendering = false;
            OVRManager.foveatedRenderingLevel = FoveationLevels[_foveation];
            Debug.Log($"[P2Pro] {FoveationLabel} (supported {OVRManager.fixedFoveatedRenderingSupported}, now {OVRManager.foveatedRenderingLevel})");
        }

        void ApplyResolution()
        {
            XRSettings.eyeTextureResolutionScale = Resolutions[_resolution];
            Debug.Log("[P2Pro] " + ResolutionLabel);
        }

        void ApplyMsaa()
        {
            if (GraphicsSettings.currentRenderPipeline is UniversalRenderPipelineAsset urp)
                urp.msaaSampleCount = MsaaSamples[_msaa];
            Debug.Log("[P2Pro] " + MsaaLabel);
        }

        void Update()
        {
            _frames++;
            _since += Time.unscaledDeltaTime;
            _worstMs = Math.Max(_worstMs, Time.unscaledDeltaTime * 1000);
            FrameTimingManager.CaptureFrameTimings();
            if (FrameTimingManager.GetLatestTimings(1, _timing) > 0)
            {
                _cpu += _timing[0].cpuMainThreadFrameTime;
                _render += _timing[0].cpuRenderThreadFrameTime;
                _gpu += _timing[0].gpuFrameTime;
                _timed++;
            }
            if (_since < Interval) return;

            float fps = _frames / _since;
            string times = _timed > 0
                ? $"CPU {_cpu / _timed:F1} ms, render {_render / _timed:F1} ms, GPU {_gpu / _timed:F1} ms"
                : "no frame timings";
            Summary = $"{fps:F0}/{Rate:F0} fps, worst {_worstMs:F0} ms, {times}";
            Debug.Log($"[P2Pro] perf: {Summary}; {FoveationLabel}, {ResolutionLabel}, {MsaaLabel}, {ScanLabel}, {PaintLabel}; {_context?.Invoke()}");
            _frames = _timed = 0;
            _cpu = _render = _gpu = _worstMs = 0;
            _since = 0;
        }

        void OnDestroy() => PlayerPrefs.Save();
    }
}
