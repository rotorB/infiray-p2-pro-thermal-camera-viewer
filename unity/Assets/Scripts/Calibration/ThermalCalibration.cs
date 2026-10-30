using System;
using System.IO;
using UnityEngine;

namespace P2Pro
{
    /// <summary>
    /// Where the P2 Pro sits on the headset and how it projects, fitted by Train mode.
    ///
    /// Thermal camera frame: x right, y up, z forward (Unity convention). Pixel coordinates:
    /// x right, y up from the bottom edge, pixel i covering [i, i+1), i.e. the layout of
    /// <see cref="ThermalCamera.Temperature"/>. A point P in the head frame projects as
    ///   c = Inverse(Rotation) * (P - Position);  x = c.x / c.z;  y = c.y / c.z;
    ///   d = 1 + K1 * (x² + y²);  u = Cx + Fx * x * d;  v = Cy + Fy * y * d.
    /// The head frame is OVRPlugin's Head node at the thermal frame's exposure time, which is
    /// the USB arrival time minus <see cref="LatencyMs"/>.
    /// </summary>
    [Serializable]
    public class ThermalCalibration
    {
        public int Version = 1;
        public string CreatedUtc;
        public Vector3 Position;          // thermal camera in the head frame, metres
        public Quaternion Rotation = Quaternion.identity;
        public float Fx, Fy, Cx, Cy, K1;
        public float LatencyMs;
        public float RmsPx;
        public int Samples;
        public int RgbViews;

        public static string FilePath => Path.Combine(Application.persistentDataPath, "thermal_calibration.json");

        /// <summary>Nominal P2 Pro intrinsics (56° x 42° field of view, 256 x 192), no mount.</summary>
        public static ThermalCalibration Nominal() => new()
        {
            Fx = 128 / Mathf.Tan(28 * Mathf.Deg2Rad),
            Fy = 96 / Mathf.Tan(21 * Mathf.Deg2Rad),
            Cx = 128,
            Cy = 96,
        };

        /// <summary>Thermal pixel of a head-frame point; false if it is behind the camera.</summary>
        public bool HeadToPixel(Vector3 head, out Vector2 pixel)
        {
            var c = Quaternion.Inverse(Rotation) * (head - Position);
            pixel = default;
            if (c.z <= 1e-4f) return false;
            float x = c.x / c.z, y = c.y / c.z;
            float d = 1 + K1 * (x * x + y * y);
            pixel = new Vector2(Cx + Fx * x * d, Cy + Fy * y * d);
            return true;
        }

        /// <summary>Thermal pixel of a world point seen with the head at <paramref name="headPose"/>.</summary>
        public bool WorldToPixel(Vector3 world, Pose headPose, out Vector2 pixel) =>
            HeadToPixel(Quaternion.Inverse(headPose.rotation) * (world - headPose.position), out pixel);

        /// <summary>Thermal camera pose in the world for a head pose.</summary>
        public Pose CameraPose(Pose headPose) =>
            new(headPose.position + headPose.rotation * Position, headPose.rotation * Rotation);

        /// <summary>
        /// Angle (degrees, counter-clockwise) to rotate the thermal image by so that the head's
        /// up direction points up, i.e. the inverse of the mount's roll.
        /// </summary>
        public float ImageRollDeg
        {
            get
            {
                var up = Quaternion.Inverse(Rotation) * Vector3.up;
                return 90 - Mathf.Atan2(Fy * up.y, Fx * up.x) * Mathf.Rad2Deg;
            }
        }

        public void Save()
        {
            CreatedUtc = DateTime.UtcNow.ToString("o");
            File.WriteAllText(FilePath, JsonUtility.ToJson(this, true));
        }

        public static ThermalCalibration Load()
        {
            try
            {
                return File.Exists(FilePath) ? JsonUtility.FromJson<ThermalCalibration>(File.ReadAllText(FilePath)) : null;
            }
            catch (Exception e)
            {
                Debug.LogWarning("[P2Pro] could not read calibration: " + e.Message);
                return null;
            }
        }

        public override string ToString()
        {
            var e = Rotation.eulerAngles;
            return $"RMS {RmsPx:F2} px over {Samples} frames ({RgbViews} RGB views), latency {LatencyMs:F0} ms\n" +
                   $"mount: x {Position.x * 100:F1} y {Position.y * 100:F1} z {Position.z * 100:F1} cm, " +
                   $"pitch {Wrap(e.x):F1} yaw {Wrap(e.y):F1} roll {Wrap(e.z):F1} deg\n" +
                   $"image roll {Mathf.DeltaAngle(0, ImageRollDeg):F1} deg, fx {Fx:F1} fy {Fy:F1} cx {Cx:F1} cy {Cy:F1} k1 {K1:F3}";
        }

        static float Wrap(float deg) => deg > 180 ? deg - 360 : deg;
    }
}
