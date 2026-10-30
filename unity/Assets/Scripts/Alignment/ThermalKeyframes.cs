using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace P2Pro
{
    /// <summary>
    /// Whole thermal frames kept for the painted room, so it shows the camera's full resolution
    /// instead of the map's texels: each texel of <see cref="ThermalMap"/> names one keyframe,
    /// and P2Pro/ThermalMesh projects that keyframe onto the surface exactly as the live view
    /// does. The keyframe named is the one that saw the texel nearest its image centre, where
    /// the sensor reads most evenly (P2ProThermalBake.compute), not simply the latest: the
    /// latest one sees a point last at its trailing edge, which painted a wall differently by
    /// sweep direction and showed the frame seams. Older keyframes count for less, so surfaces
    /// still update.
    /// A keyframe is never changed once stored, so every texel it names stays inside it. A new
    /// one is taken after a small head move or turn, or once a second while still (fresh °C),
    /// at most <see cref="MaxPerSecond"/> a second. Its slot is the one fewest texels still name
    /// (counted twice a second by the map's histogram pass); a keyframe whose surfaces have all
    /// been seen again by newer ones is free. Texels whose keyframe was replaced fall back to
    /// the map's own °C.
    /// Tag in the map: 1 + slot + Capacity * generation, exact in half floats; 0 = none.
    /// </summary>
    public class ThermalKeyframes
    {
        public const int Capacity = 128;
        const int Generations = 15;             // 1 + 127 + 128 * 14 < 2048
        const float NewAfterMetres = 0.03f;
        const float NewAfterDegrees = 2;
        const float RefreshSeconds = 1;
        const float MaxPerSecond = 5;

        static readonly int FramesId = Shader.PropertyToID("_KeyframeTex");
        static readonly int ParamsId = Shader.PropertyToID("_Keyframes");
        static readonly int RgbAspectId = Shader.PropertyToID("_RgbAspect");
        static readonly int ThermalAspectId = Shader.PropertyToID("_ThermalAspect");

        // As struct Keyframe in P2ProThermalMesh.shader and P2ProThermalBake.compute.
        [StructLayout(LayoutKind.Sequential)]
        struct Params
        {
            public Vector4 Row0, Row1, Row2;    // world -> RGB camera
            public Vector4 Position;            // RGB camera; w = tag, 0 = empty slot
            public Vector4 Intrinsics, Crop, MapA, MapB;
            public Vector4 Info;                // x = time taken (s, unscaled)
        }

        readonly Params[] _params = new Params[Capacity];
        readonly int[] _generation = new int[Capacity];
        readonly uint[] _use = new uint[Capacity];
        readonly int[] _takenAt = new int[Capacity];
        readonly ComputeBuffer _buffer = new(Capacity, Marshal.SizeOf<Params>());
        RenderTexture _frames;
        int _taken;                             // keyframes taken so far
        int _counted;                           // _use covers keyframes taken before this
        int _current = -1;
        Pose _last;
        float _lastTime = float.NegativeInfinity;

        /// <summary>Per-slot parameters, for the map's usage count.</summary>
        public ComputeBuffer Buffer => _buffer;

        public ThermalKeyframes()
        {
            for (int i = 0; i < Capacity; i++) _takenAt[i] = -1;
            Create(1, 1);                       // something to bind until the first frame
        }

        /// <summary>
        /// Keeps the latest thermal frame as a keyframe when it is due; returns its tag, or -1
        /// when the frame is not kept (texels keep the keyframe they name).
        /// </summary>
        public float Add(Texture temperature, RgbProjection projection)
        {
            if (_frames.width != temperature.width || _frames.height != temperature.height)
                Create(temperature.width, temperature.height);

            float now = Time.unscaledTime;
            var pose = new Pose(projection.RgbPosition, projection.WorldToRgb.inverse.rotation);
            bool due = _current < 0 || now - _lastTime > RefreshSeconds ||
                       Vector3.Distance(pose.position, _last.position) > NewAfterMetres ||
                       Quaternion.Angle(pose.rotation, _last.rotation) > NewAfterDegrees;
            if (!due || now - _lastTime < 1 / MaxPerSecond) return -1;

            int slot = FreeSlot();
            _generation[slot] = (_generation[slot] + 1) % Generations;
            _takenAt[slot] = _taken++;
            _current = slot;
            _last = pose;
            _lastTime = now;

            Graphics.Blit(temperature, _frames, 0, slot);
            var m = projection.WorldToRgb;
            float tag = 1 + slot + Capacity * _generation[slot];
            _params[slot] = new Params
            {
                Row0 = m.GetRow(0), Row1 = m.GetRow(1), Row2 = m.GetRow(2),
                Position = new Vector4(projection.RgbPosition.x, projection.RgbPosition.y, projection.RgbPosition.z, tag),
                Intrinsics = projection.Intrinsics, Crop = projection.Crop, MapA = projection.MapA, MapB = projection.MapB,
                Info = new Vector4(now, 0, 0, 0),
            };
            _buffer.SetData(_params, slot, slot, 1);
            return tag;
        }

        // An empty slot, else the one fewest texels name; slots taken since the last count
        // have no count yet and are kept.
        int FreeSlot()
        {
            int best = -1;
            uint bestUse = uint.MaxValue;
            for (int i = 0; i < Capacity; i++)
            {
                if (_params[i].Position.w == 0) return i;
                if (i == _current || _takenAt[i] >= _counted) continue;
                if (_use[i] < bestUse) { best = i; bestUse = _use[i]; }
            }
            // Every slot taken since the last count: the oldest.
            if (best < 0)
                for (int i = 0; i < Capacity; i++)
                    if (i != _current && (best < 0 || _takenAt[i] < _takenAt[best])) best = i;
            return best;
        }

        /// <summary>Marks the start of a usage count (the map's histogram pass).</summary>
        public int BeginCount() => _taken;

        /// <summary>Texels naming each slot, counted for keyframes taken before <paramref name="stamp"/>.</summary>
        public void SetUse(Unity.Collections.NativeArray<uint> use, int stamp)
        {
            use.CopyTo(_use);
            _counted = stamp;
        }

        /// <summary>Forgets all keyframes; texels naming them fall back to the map's °C.</summary>
        public void Clear()
        {
            for (int i = 0; i < Capacity; i++) _params[i].Position.w = 0;
            _buffer.SetData(_params);
            _current = -1;
        }

        public void Bind(Material material)
        {
            material.SetTexture(FramesId, _frames);
            material.SetBuffer(ParamsId, _buffer);
            material.SetFloat(RgbAspectId, ThermalAlignment.RgbAspect);
            material.SetFloat(ThermalAspectId, ThermalAlignment.ThermalAspect);
        }

        // A new frame size (sharpening switched) makes the stored ones unusable. Half floats:
        // 0.016 °C steps at 30 °C.
        void Create(int width, int height)
        {
            if (_frames != null) _frames.Release();
            _frames = new RenderTexture(width, height, 0, GraphicsFormat.R16_SFloat)
            {
                name = "Thermal keyframes",
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = Capacity,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
            };
            _frames.Create();
            Clear();
            if (width > 1)
                Debug.Log($"[P2Pro] thermal keyframes: {Capacity} x {width}x{height} R16F = {Capacity * width * height * 2 / 1048576} MB");
        }

        public void Dispose()
        {
            if (_frames != null) _frames.Release();
            _frames = null;
            _buffer.Release();
        }
    }
}
