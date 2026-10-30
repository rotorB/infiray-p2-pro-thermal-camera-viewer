using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Android;

namespace P2Pro
{
    /// <summary>
    /// Infiray P2 Pro on the headset's USB-C port. Asks for the CAMERA and USB_CAMERA
    /// permissions, lets the Android plugin (android/unityplugin) open the device, then streams
    /// through libp2prouvc and keeps <see cref="Temperature"/>, a 256x192 RFloat texture in °C,
    /// up to date. Texture row 0 is the bottom of the image.
    /// Each frame carries its CLOCK_MONOTONIC arrival time (<see cref="TimeNs"/>, the XrTime
    /// base) and the head pose at that time, for pairing with the headset cameras.
    /// The camera's automatic shutter (its own flat-field calibration, every 15-60 s) froze the
    /// picture for ~0.5 s and the frozen copies arrived with fresh times and head poses, so the
    /// picture slid across the room. It can be switched off from the Setup tab
    /// (<see cref="AutoShutter"/>), at the cost of temperatures drifting as the sensor warms; it
    /// stays on by default. Nothing here ever triggers a calibration. Repeated frames are still counted, to show any shutter that happens.
    /// </summary>
    public class ThermalCamera : MonoBehaviour
    {
        public const int Width = 256;
        public const int Height = 192;
        const int Fps = 25;
        const string UsbCameraPermission = "horizonos.permission.USB_CAMERA";
        const float StatusPollSeconds = 0.25f;

        // The camera's YUYV frame is Width x 2*Height: the 8-bit image on top, temperatures below.
        [DllImport("p2prouvc")] static extern int p2pro_open(int fd, int width, int height, int fps);
        [DllImport("p2prouvc")] static extern int p2pro_take_celsius([Out] float[] dst, int width, int height, [Out] float[] stats, out long timeNs);
        [DllImport("p2prouvc")] static extern void p2pro_close();
        [DllImport("p2prouvc")] public static extern long p2pro_now_ns();
        [DllImport("p2prouvc")] static extern int p2pro_set_auto_shutter(int on);

        public Texture2D Temperature { get; private set; }
        public float Min { get; private set; }
        public float Max { get; private set; }
        public float Center { get; private set; }
        public int FrameCount { get; private set; }
        /// <summary>CLOCK_MONOTONIC ns when the latest frame finished arriving over USB.</summary>
        public long TimeNs { get; private set; }
        /// <summary>Head pose (tracking space) at <see cref="TimeNs"/>.</summary>
        public Pose HeadPose { get; private set; }
        /// <summary>Latest frame in °C, same layout as <see cref="Temperature"/>. Overwritten on the next frame.</summary>
        public float[] Celsius => _pixels;
        public string Status { get; private set; } = "Waiting for permissions";
        /// <summary>Keeps the last frame in <see cref="Temperature"/> and <see cref="Celsius"/>; new frames are skipped.</summary>
        public bool Paused { get; set; }

        /// <summary>The camera repeats its last frame: the shutter is closed.</summary>
        public bool ShutterActive => _repeats > 0;
        /// <summary>Shutter closings seen since start.</summary>
        public int ShutterCount { get; private set; }
        /// <summary>
        /// The camera's own periodic calibration: on by default (remembered). Written to the
        /// camera after its startup calibrations on every stream start (the camera forgets it
        /// when it loses power) and when switched.
        /// </summary>
        public bool AutoShutter { get; private set; }
        public string AutoShutterLabel => $"Camera shutter: {(AutoShutter ? "auto" : "off")}{_autoShutterState}";

        /// <summary>Raised on the main thread after <see cref="Temperature"/> got a new frame.</summary>
        public event Action FrameReady;

        const string AutoShutterKey = "p2pro.autoShutter2";      // v1 defaulted to off
        const float StartupCalibrationSeconds = 12;   // the camera's two startup shutters come ~5 and ~9 s after streaming starts
        const float AutoShutterRetrySeconds = 2;
        const int AutoShutterAttempts = 5;
        const int RepeatStride = 7;             // pixels compared, one in seven
        const float RepeatShare = 0.98f;        // unchanged share that marks a repeated frame
        float[] _pixels = new float[Width * Height];
        float[] _incoming = new float[Width * Height];
        int _repeats;
        float _shutterStartedAt;
        float _autoShutterAt = -1;
        int _autoShutterTries;
        string _autoShutterState = "";
        volatile bool _autoShutterBusy;
        volatile int _autoShutterResult = int.MinValue;
        readonly float[] _stats = new float[3];
        AndroidJavaClass _usb;
        int _openGeneration;
        bool _streaming;
        float _nextStatusPoll;
        float _nextLog;
        int _framesAtLastLog;

        void Start()
        {
            Temperature = new Texture2D(Width, Height, TextureFormat.RFloat, false, true)
            {
                name = "P2Pro temperature",
                wrapMode = TextureWrapMode.Clamp,
                filterMode = SystemInfo.IsFormatSupported(UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat,
                    UnityEngine.Experimental.Rendering.GraphicsFormatUsage.Linear)
                    ? FilterMode.Bilinear : FilterMode.Point,
            };
            AutoShutter = PlayerPrefs.GetInt(AutoShutterKey, 1) == 1;
            RequestPermissions();
        }

        /// <summary>Switches the camera's own periodic calibration on or off.</summary>
        public void ToggleAutoShutter()
        {
            AutoShutter = !AutoShutter;
            PlayerPrefs.SetInt(AutoShutterKey, AutoShutter ? 1 : 0);
            PlayerPrefs.Save();
            if (!_streaming) return;
            _autoShutterTries = 0;
            if (_autoShutterAt < 0) _autoShutterAt = Time.unscaledTime;   // else the startup write comes anyway
        }

        // The command blocks for up to a few seconds, so it runs on a worker thread.
        void UpdateAutoShutter()
        {
            int result = _autoShutterResult;
            if (result != int.MinValue)
            {
                _autoShutterResult = int.MinValue;
                int want = AutoShutter ? 1 : 0;
                if (result == want)
                {
                    _autoShutterState = "";
                    Debug.Log($"[P2Pro] camera auto-shutter {(AutoShutter ? "on" : "off")} (read back {result})");
                }
                else if (++_autoShutterTries < AutoShutterAttempts)
                {
                    _autoShutterState = " (retrying)";
                    _autoShutterAt = Time.unscaledTime + AutoShutterRetrySeconds;
                    Debug.Log($"[P2Pro] camera auto-shutter := {want}: got {result}, retrying");
                }
                else
                {
                    _autoShutterState = $" (camera refused: {result})";
                    Debug.Log($"[P2Pro] camera auto-shutter := {want}: gave up, last result {result}");
                }
            }
            if (_autoShutterAt < 0 || Time.unscaledTime < _autoShutterAt || _autoShutterBusy) return;
            _autoShutterAt = -1;
            _autoShutterBusy = true;
            _autoShutterState = " (setting)";
            int on = AutoShutter ? 1 : 0;
            System.Threading.Tasks.Task.Run(() =>
            {
                try { _autoShutterResult = p2pro_set_auto_shutter(on); }
                catch (Exception) { _autoShutterResult = -1; }
                finally { _autoShutterBusy = false; }
            });
        }

        void RequestPermissions()
        {
            var missing = new List<string>();
            foreach (var p in new[] { Permission.Camera, UsbCameraPermission })
                if (!Permission.HasUserAuthorizedPermission(p)) missing.Add(p);
            if (missing.Count == 0)
            {
                StartUsb();
                return;
            }

            int pending = missing.Count;
            var callbacks = new PermissionCallbacks();
            void Answered(string _)
            {
                if (--pending > 0) return;
                if (Permission.HasUserAuthorizedPermission(Permission.Camera)) StartUsb();
                else SetStatus("Camera permission denied: the P2 Pro cannot be opened");
            }
            callbacks.PermissionGranted += Answered;
            callbacks.PermissionDenied += Answered;
            Permission.RequestUserPermissions(missing.ToArray(), callbacks);
        }

        void StartUsb()
        {
            _usb = new AndroidJavaClass("com.p2pro.quest.unity.UsbThermal");
            using var player = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
            using var activity = player.GetStatic<AndroidJavaObject>("currentActivity");
            _usb.CallStatic("start", activity);
        }

        void Update()
        {
            if (_usb == null) return;

            if (Time.unscaledTime >= _nextStatusPoll)
            {
                _nextStatusPoll = Time.unscaledTime + StatusPollSeconds;
                int generation = _usb.CallStatic<int>("getGeneration");
                if (generation != _openGeneration)
                {
                    _openGeneration = generation;
                    OpenStream(_usb.CallStatic<int>("getFd"));
                }
                if (!_streaming) SetStatus(_usb.CallStatic<string>("getStatus"));
            }

            if (!_streaming) return;
            UpdateAutoShutter();
            if (Paused) return;
            int result = p2pro_take_celsius(_incoming, Width, Height, _stats, out long timeNs);
            if (result < 0)
            {
                CloseStream();
                SetStatus("P2 Pro stream stopped");
                return;
            }
            if (result == 0) return;
            if (IsRepeat())
            {
                if (_repeats++ == 0)
                {
                    _shutterStartedAt = Time.unscaledTime;
                    ShutterCount++;
                }
            }
            else if (_repeats > 0)
            {
                Debug.Log($"[P2Pro] shutter: camera repeated its frame {_repeats} times over " +
                          $"{Time.unscaledTime - _shutterStartedAt:F2} s, closing {ShutterCount}, auto-shutter {(AutoShutter ? "on" : "off")}");
                _repeats = 0;
            }
            (_pixels, _incoming) = (_incoming, _pixels);

            Temperature.SetPixelData(_pixels, 0);
            Temperature.Apply(false);
            Min = _stats[0];
            Max = _stats[1];
            Center = _stats[2];
            TimeNs = timeNs;
            HeadPose = HeadPoseAt(timeNs);
            FrameCount++;
            FrameReady?.Invoke();
            LogStats();
        }

        /// <summary>Head pose in tracking space at a CLOCK_MONOTONIC / XrTime instant.</summary>
        public static Pose HeadPoseAt(long timeNs)
        {
            var pose = OVRPlugin.GetNodePoseStateAtTime(timeNs * 1e-9, OVRPlugin.Node.Head).Pose.ToOVRPose();
            return new Pose(pose.position, pose.orientation);
        }

        void OpenStream(int fd)
        {
            CloseStream();
            if (fd < 0) return;
            int rc = p2pro_open(fd, Width, Height * 2, Fps);
            _streaming = rc == 0;
            if (_streaming)
            {
                _autoShutterAt = Time.unscaledTime + StartupCalibrationSeconds;
                _autoShutterTries = 0;
                _autoShutterState = " (after startup calibration)";
            }
            SetStatus(_streaming ? "Streaming" : $"Could not start the P2 Pro stream (error {rc})");
        }

        void CloseStream()
        {
            if (!_streaming) return;
            p2pro_close();
            _streaming = false;
        }

        // The new frame (_incoming) equals the last one (_pixels) almost everywhere. Values
        // come in 1/64 K steps; a live sensor changes most pixels by a step or more each frame.
        bool IsRepeat()
        {
            if (FrameCount == 0) return false;
            int same = 0, n = 0;
            for (int i = 0; i < _pixels.Length; i += RepeatStride, n++)
                if (Mathf.Abs(_incoming[i] - _pixels[i]) < 0.001f) same++;
            return (float)same / n >= RepeatShare;
        }

        void LogStats()
        {
            if (Time.unscaledTime < _nextLog) return;
            float fps = (FrameCount - _framesAtLastLog) / 2f;
            _framesAtLastLog = FrameCount;
            _nextLog = Time.unscaledTime + 2f;
            Debug.Log($"[P2Pro] thermal {fps:F0} fps, min {Min:F1} max {Max:F1} center {Center:F1} °C");
        }

        void SetStatus(string status)
        {
            if (status == Status) return;
            Status = status;
            Debug.Log("[P2Pro] thermal: " + status);
        }

        void OnDestroy()
        {
            CloseStream();
            _usb?.CallStatic("stop");
            _usb?.Dispose();
        }
    }
}
