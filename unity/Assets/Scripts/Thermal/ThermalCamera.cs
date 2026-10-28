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
        [DllImport("p2prouvc")] static extern int p2pro_take_celsius([Out] float[] dst, int width, int height, [Out] float[] stats);
        [DllImport("p2prouvc")] static extern void p2pro_close();

        public Texture2D Temperature { get; private set; }
        public float Min { get; private set; }
        public float Max { get; private set; }
        public float Center { get; private set; }
        public int FrameCount { get; private set; }
        public string Status { get; private set; } = "Waiting for permissions";

        /// <summary>Raised on the main thread after <see cref="Temperature"/> got a new frame.</summary>
        public event Action FrameReady;

        readonly float[] _pixels = new float[Width * Height];
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
            RequestPermissions();
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
            int result = p2pro_take_celsius(_pixels, Width, Height, _stats);
            if (result < 0)
            {
                CloseStream();
                SetStatus("P2 Pro stream stopped");
                return;
            }
            if (result == 0) return;

            Temperature.SetPixelData(_pixels, 0);
            Temperature.Apply(false);
            Min = _stats[0];
            Max = _stats[1];
            Center = _stats[2];
            FrameCount++;
            FrameReady?.Invoke();
            LogStats();
        }

        void OpenStream(int fd)
        {
            CloseStream();
            if (fd < 0) return;
            int rc = p2pro_open(fd, Width, Height * 2, Fps);
            _streaming = rc == 0;
            SetStatus(_streaming ? "Streaming" : $"Could not start the P2 Pro stream (error {rc})");
        }

        void CloseStream()
        {
            if (!_streaming) return;
            p2pro_close();
            _streaming = false;
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
