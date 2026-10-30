using Genesis.RoomScan;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Shader parameters that take a world point to the thermal image of the latest frame: the
    /// headset RGB camera as it stood when the frame was exposed (pose, intrinsics, crop), then
    /// the <see cref="ThermalAlignment"/> lens model, or its mapping as A + B / distance.
    /// Consumed by P2ProThermalBake.compute and P2Pro/ThermalLive under the same names
    /// (P2ProThermalLens.hlsl).
    /// </summary>
    public struct RgbProjection
    {
        static readonly int WorldToRgbId = Shader.PropertyToID("_WorldToRgb");
        static readonly int RgbPositionId = Shader.PropertyToID("_RgbPosition");
        static readonly int RgbIntrinsicsId = Shader.PropertyToID("_RgbIntrinsics");
        static readonly int RgbCropId = Shader.PropertyToID("_RgbCrop");
        static readonly int MapAId = Shader.PropertyToID("_MapA");
        static readonly int MapBId = Shader.PropertyToID("_MapB");
        static readonly int RgbAspectId = Shader.PropertyToID("_RgbAspect");
        static readonly int ThermalAspectId = Shader.PropertyToID("_ThermalAspect");

        public Matrix4x4 WorldToRgb;
        public Vector3 RgbPosition;
        public Vector4 Intrinsics;              // fx, fy, cx, cy in sensor pixels
        public Vector4 Crop;                    // image rectangle within the sensor
        public Vector4 MapA, MapB;              // roll in radians
        public ThermalLens Lens;

        /// <summary>False without an RGB camera image, a thermal frame or an alignment.</summary>
        public static bool TryCreate(ThermalPanel panel, ThermalAlignment alignment, LatencyEstimator latency,
            out RgbProjection projection)
        {
            projection = default;
            var source = panel.Rgb.Source;
            if (source == null || !source.IsPlaying || panel.Thermal.FrameCount == 0) return false;
            // With only the lens model, the mapping is unused (zero).
            if (!alignment.TryGetCoefficients(out var a, out var b) && !alignment.Lens.Solved) return false;
            a.w *= Mathf.Deg2Rad;
            b.w *= Mathf.Deg2Rad;

            var pose = FramePose(panel, latency);

            var intrinsics = source.Intrinsics;
            var sensor = (Vector2)intrinsics.SensorResolution;
            var crop = (Vector2)source.CurrentResolution / sensor;
            crop /= Mathf.Max(crop.x, crop.y);

            projection = new RgbProjection
            {
                WorldToRgb = Matrix4x4.TRS(pose.position, pose.rotation, Vector3.one).inverse,
                RgbPosition = pose.position,
                Intrinsics = new Vector4(intrinsics.FocalLength.x, intrinsics.FocalLength.y,
                    intrinsics.PrincipalPoint.x, intrinsics.PrincipalPoint.y),
                Crop = new Vector4(sensor.x * (1 - crop.x) / 2, sensor.y * (1 - crop.y) / 2, sensor.x * crop.x, sensor.y * crop.y),
                MapA = a,
                MapB = b,
                Lens = alignment.Lens,
            };
            return true;
        }

        /// <summary>World pose of the headset RGB camera when the latest thermal frame was exposed.</summary>
        public static Pose FramePose(ThermalPanel panel, LatencyEstimator latency)
        {
            var pose = CameraPoseAt(panel.Thermal.TimeNs - (long)(latency.LatencyMs * 1e6f), panel.Rgb.Source);
            var depth = DepthCapture.Instance;
            return depth != null ? depth.TrackingToWorld(pose) : pose;
        }

        /// <summary>Headset RGB camera pose (tracking space) at a CLOCK_MONOTONIC instant.</summary>
        public static Pose CameraPoseAt(long timeNs, Meta.XR.PassthroughCameraAccess source)
        {
            var head = ThermalCamera.HeadPoseAt(timeNs);
            var lens = source.Intrinsics.LensOffset;
            return new Pose(head.position + head.rotation * lens.position, head.rotation * lens.rotation);
        }

        public void Apply(ComputeShader shader)
        {
            shader.SetMatrix(WorldToRgbId, WorldToRgb);
            shader.SetVector(RgbPositionId, RgbPosition);
            shader.SetVector(RgbIntrinsicsId, Intrinsics);
            shader.SetVector(RgbCropId, Crop);
            shader.SetVector(MapAId, MapA);
            shader.SetVector(MapBId, MapB);
            shader.SetFloat(RgbAspectId, ThermalAlignment.RgbAspect);
            shader.SetFloat(ThermalAspectId, ThermalAlignment.ThermalAspect);
            Lens.Apply(shader);
        }

        public void Apply(Material material)
        {
            material.SetMatrix(WorldToRgbId, WorldToRgb);
            material.SetVector(RgbPositionId, RgbPosition);
            material.SetVector(RgbIntrinsicsId, Intrinsics);
            material.SetVector(RgbCropId, Crop);
            material.SetVector(MapAId, MapA);
            material.SetVector(MapBId, MapB);
            material.SetFloat(RgbAspectId, ThermalAlignment.RgbAspect);
            material.SetFloat(ThermalAspectId, ThermalAlignment.ThermalAspect);
            Lens.Apply(material);
        }
    }
}
