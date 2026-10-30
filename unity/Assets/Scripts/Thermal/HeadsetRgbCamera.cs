using System;
using System.Reflection;
using Meta.XR;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// The headset's passthrough RGB camera, read through the scene's
    /// <see cref="PassthroughCameraAccess"/> (the one QuestRoomScan already runs, left camera,
    /// 1280x960). Reports every new image with its exposure time on the CLOCK_MONOTONIC / XrTime
    /// base shared with <see cref="ThermalCamera.TimeNs"/>, the camera pose at that time and the
    /// lens intrinsics.
    /// </summary>
    public class HeadsetRgbCamera : MonoBehaviour
    {
        // PassthroughCameraAccess only exposes a wall-clock DateTime; the monotonic time it uses
        // for its own pose lookup is private.
        static readonly FieldInfo MonotonicField =
            typeof(PassthroughCameraAccess).GetField("_timestampNsMonotonic", BindingFlags.Instance | BindingFlags.NonPublic);

        public PassthroughCameraAccess Source { get; private set; }
        public Texture Texture => Source != null && Source.IsPlaying ? Source.GetTexture() : null;
        public long TimeNs { get; private set; }
        /// <summary>Camera (lens) pose at <see cref="TimeNs"/>, world space (PassthroughCameraAccess.GetCameraPose).</summary>
        public Pose CameraPose { get; private set; }
        public PassthroughCameraAccess.CameraIntrinsics Intrinsics { get; private set; }
        public int FrameCount { get; private set; }

        /// <summary>Raised after a new RGB image arrived.</summary>
        public event Action FrameReady;

        void Start()
        {
            if (MonotonicField == null)
                Debug.LogError("[P2Pro] PassthroughCameraAccess._timestampNsMonotonic not found: RGB frames get no time");
        }

        // LateUpdate: PassthroughCameraAccess fetches the new image in its own Update.
        void LateUpdate()
        {
            if (Source == null)
            {
                Source = FindAnyObjectByType<PassthroughCameraAccess>();
                if (Source == null) return;
                Debug.Log($"[P2Pro] RGB camera: {Source.CameraPosition}, requested {Source.RequestedResolution}");
            }
            if (!Source.IsPlaying || MonotonicField == null) return;

            long timeNs = (long)MonotonicField.GetValue(Source);
            if (timeNs == 0 || timeNs == TimeNs) return;

            TimeNs = timeNs;
            CameraPose = Source.GetCameraPose();
            Intrinsics = Source.Intrinsics;
            FrameCount++;
            FrameReady?.Invoke();
        }
    }
}
