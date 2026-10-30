using Genesis.RoomScan;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace P2Pro
{
    /// <summary>
    /// The room's persistent thermal texture: every thermal frame is projected onto the surfaces
    /// the environment depth sees (P2ProThermalBake.compute) and stored as °C in three world-space
    /// maps; the room mesh draws them through the palette (P2Pro/ThermalMesh). Surfaces in view
    /// keep updating, surfaces out of view keep their last temperature. Twice a second a histogram
    /// of all stored temperatures gives the Scene palette range (2nd to 98th percentile).
    /// The map's texels are coarse for a whole home; each also names the keyframe that last saw
    /// it (<see cref="ThermalKeyframes"/>), which the mesh projects at the camera's resolution.
    /// Projection: see <see cref="RgbProjection"/>.
    /// </summary>
    public class ThermalMap : MonoBehaviour
    {
        const int MapSize = 2048;               // per axis pair; 19.2 m volume -> 9 mm x 19 mm texels
        const float MaxDistance = 5;
        const int HistogramBins = 960;          // as in P2ProThermalBake.compute: -40 °C up, 0.25 °C steps
        const float HistogramMin = -40, HistogramStep = 0.25f;
        const float SceneLow = 0.02f, SceneHigh = 0.98f;
        const float SceneInterval = 0.5f;

        static readonly int MapXZId = Shader.PropertyToID("_MapXZ");
        static readonly int MapXYId = Shader.PropertyToID("_MapXY");
        static readonly int MapYZId = Shader.PropertyToID("_MapYZ");
        static readonly int TagXZId = Shader.PropertyToID("_TagXZ");
        static readonly int TagXYId = Shader.PropertyToID("_TagXY");
        static readonly int TagYZId = Shader.PropertyToID("_TagYZ");
        static readonly int MapSizeId = Shader.PropertyToID("_MapSize");
        static readonly int TemperatureId = Shader.PropertyToID("_Temperature");
        static readonly int FlatSumId = Shader.PropertyToID("_FlatSum");
        static readonly int FlatCountId = Shader.PropertyToID("_FlatCount");
        static readonly int FlatLearnId = Shader.PropertyToID("_FlatLearn");
        static readonly int MaxDistanceId = Shader.PropertyToID("_MaxDistance");
        static readonly int PaletteId = Shader.PropertyToID("_Palette");
        static readonly int MinId = Shader.PropertyToID("_Min");
        static readonly int MaxId = Shader.PropertyToID("_Max");
        static readonly int HistogramId = Shader.PropertyToID("_Histogram");
        static readonly int SamplePointId = Shader.PropertyToID("_SamplePoint");
        static readonly int SampleResultId = Shader.PropertyToID("_SampleResult");
        static readonly int VoxCountId = Shader.PropertyToID("gsVoxCount");
        static readonly int VoxSizeId = Shader.PropertyToID("gsVoxSize");
        static readonly int KeyframeTagId = Shader.PropertyToID("_KeyframeTag");
        static readonly int KeyframesId = Shader.PropertyToID("_Keyframes");
        static readonly int KeyframeUseId = Shader.PropertyToID("_KeyframeUse");
        static readonly int NowId = Shader.PropertyToID("_Now");

        ThermalPanel _panel;
        ThermalAlignment _alignment;
        LatencyEstimator _latency;
        OverlayStyle _style;
        ComputeShader _bake;
        int _bakeKernel, _clearKernel, _clearHistogramKernel, _histogramKernel, _sampleKernel;
        ComputeBuffer _histogram;
        ComputeBuffer _sample;
        ComputeBuffer _keyframeUse;
        bool _histogramReadback;
        bool _useReadback;
        bool _sampleReadback;
        float _nextScene;
        int _clears;
        RenderTexture _xz, _xy, _yz;
        RenderTexture _tagXz, _tagXy, _tagYz;    // up to four keyframes per texel
        ThermalKeyframes _keyframes;
        Material _material;
        bool _visible;

        /// <summary>Thermal frames projected into the map so far.</summary>
        public int Baked { get; private set; }
        /// <summary>Why the last frame was not baked, or empty.</summary>
        public string Skipped { get; private set; } = "";

        /// <summary>Paints new thermal frames into the map.</summary>
        public bool Baking { get; set; }

        /// <summary>2nd and 98th percentile of the °C held in the map, or null while it is empty.</summary>
        public Vector2? SceneRange { get; private set; }

        /// <summary>Draws the map on the room mesh.</summary>
        public bool Visible
        {
            get => _visible;
            set
            {
                _visible = value;
                GPUMeshRenderer.OverlayMaterials.Remove(_material);
                if (value) GPUMeshRenderer.OverlayMaterials.Add(_material);
            }
        }

        public void Init(ThermalPanel panel, ThermalAlignment alignment, LatencyEstimator latency, OverlayStyle style)
        {
            _panel = panel;
            _style = style;
            _alignment = alignment;
            _latency = latency;
            _bake = Resources.Load<ComputeShader>("P2ProThermalBake");
            _bakeKernel = _bake.FindKernel("Bake");
            _clearKernel = _bake.FindKernel("Clear");
            _clearHistogramKernel = _bake.FindKernel("ClearHistogram");
            _histogramKernel = _bake.FindKernel("Histogram");
            _sampleKernel = _bake.FindKernel("Sample");
            _sample = new ComputeBuffer(2, 4);
            _bake.SetBuffer(_sampleKernel, SampleResultId, _sample);
            _histogram = new ComputeBuffer(HistogramBins, 4);
            _bake.SetBuffer(_clearHistogramKernel, HistogramId, _histogram);
            _bake.SetBuffer(_histogramKernel, HistogramId, _histogram);
            _xz = CreateMap("Thermal map XZ");
            _xy = CreateMap("Thermal map XY");
            _yz = CreateMap("Thermal map YZ");
            _tagXz = CreateMap("Thermal tags XZ");
            _tagXy = CreateMap("Thermal tags XY");
            _tagYz = CreateMap("Thermal tags YZ");
            _keyframes = new ThermalKeyframes();
            _keyframeUse = new ComputeBuffer(ThermalKeyframes.Capacity, 4);
            _bake.SetBuffer(_clearHistogramKernel, KeyframeUseId, _keyframeUse);
            _bake.SetBuffer(_histogramKernel, KeyframeUseId, _keyframeUse);
            _bake.SetBuffer(_histogramKernel, KeyframesId, _keyframes.Buffer);
            _bake.SetBuffer(_bakeKernel, KeyframesId, _keyframes.Buffer);
            foreach (int kernel in new[] { _bakeKernel, _clearKernel, _histogramKernel, _sampleKernel })
            {
                _bake.SetTexture(kernel, MapXZId, _xz);
                _bake.SetTexture(kernel, MapXYId, _xy);
                _bake.SetTexture(kernel, MapYZId, _yz);
                if (kernel == _sampleKernel) continue;
                _bake.SetTexture(kernel, TagXZId, _tagXz);
                _bake.SetTexture(kernel, TagXYId, _tagXy);
                _bake.SetTexture(kernel, TagYZId, _tagYz);
            }
            _bake.SetInts(MapSizeId, MapSize, MapSize);
            _bake.SetBuffer(_bakeKernel, FlatSumId, _panel.FlatField.Sum);
            _bake.SetBuffer(_bakeKernel, FlatCountId, _panel.FlatField.Count);
            _bake.SetFloat(MaxDistanceId, MaxDistance);
            Clear();

            _material = new Material(Resources.Load<Shader>("P2ProThermalMesh"));
            _material.SetTexture(MapXZId, _xz);
            _material.SetTexture(MapXYId, _xy);
            _material.SetTexture(MapYZId, _yz);
            _material.SetTexture(TagXZId, _tagXz);
            _material.SetTexture(TagXYId, _tagXy);
            _material.SetTexture(TagYZId, _tagYz);
            _panel.Thermal.FrameReady += OnThermal;
            Debug.Log($"[P2Pro] thermal map: 6 x {MapSize}x{MapSize} RGBA16F = {6 * MapSize * MapSize * 8 / 1048576} MB");
        }

        static RenderTexture CreateMap(string name)
        {
            var rt = new RenderTexture(MapSize, MapSize, 0, GraphicsFormat.R16G16B16A16_SFloat)
            {
                name = name,
                enableRandomWrite = true,
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
            };
            rt.Create();
            return rt;
        }

        public void Clear()
        {
            _bake.Dispatch(_clearKernel, MapSize / 8, MapSize / 8, 1);
            _keyframes.Clear();
            SceneRange = null;
            Baked = 0;
            _clears++;
            Debug.Log("[P2Pro] thermal map cleared");
        }

        void OnThermal()
        {
            if (!Baking) return;
            var source = _panel.Rgb.Source;
            var depth = DepthCapture.Instance;
            var volume = VolumeIntegrator.Instance;
            if (source == null || !source.IsPlaying) { Skipped = "no RGB camera"; return; }
            if (depth == null || !DepthCapture.DepthAvailable || depth.DepthTex == null || depth.NormTex == null) { Skipped = "no room depth"; return; }
            if (volume == null) { Skipped = "no scan volume"; return; }
            if (!RgbProjection.TryCreate(_panel, _alignment, _latency, out var projection)) { Skipped = "not calibrated"; return; }
            Skipped = "";

            _bake.SetMatrixArray(DepthCapture.ViewID, depth.View);
            _bake.SetMatrixArray(DepthCapture.ProjID, depth.Proj);
            _bake.SetMatrixArray(DepthCapture.ViewInvID, depth.ViewInv);
            _bake.SetMatrixArray(DepthCapture.ProjInvID, depth.ProjInv);
            _bake.SetVector(DepthCapture.ZParamsID, depth.Planes);
            _bake.SetInts(DepthCapture.TexSizeID, depth.DepthTex.width, depth.DepthTex.height);
            _bake.SetTexture(_bakeKernel, DepthCapture.DepthTexID, depth.DepthTex);
            _bake.SetTexture(_bakeKernel, DepthCapture.NormTexID, depth.NormTex);
            BindVolume(volume);
            volume.BindExclusionUniforms(_bake);
            volume.BindScanPriors(_bake);

            _bake.SetTexture(_bakeKernel, TemperatureId, _panel.Temperature);
            projection.Apply(_bake);
            _bake.SetFloat(KeyframeTagId, _keyframes.Add(_panel.Temperature, projection));
            _bake.SetFloat(NowId, Time.unscaledTime);
            _bake.SetFloat(FlatLearnId, _panel.FlatField.Enabled ? 1 : 0);
            _panel.FlatField.Apply(_bake, _bakeKernel);
            _bake.Dispatch(_bakeKernel, (depth.DepthTex.width + 7) / 8, (depth.DepthTex.height + 7) / 8, 1);
            _panel.FlatField.Learn();
            Baked++;
        }

        void BindVolume(VolumeIntegrator volume)
        {
            _bake.SetInts(VoxCountId, volume.VoxelCount.x, volume.VoxelCount.y, volume.VoxelCount.z);
            _bake.SetFloat(VoxSizeId, volume.VoxelSize);
        }

        /// <summary>
        /// Starts reading the stored °C at a world point (small neighbourhood on that surface);
        /// <paramref name="done"/> gets null where the map has no data. False while busy.
        /// </summary>
        public bool RequestTemperature(Vector3 world, System.Action<float?> done)
        {
            var volume = VolumeIntegrator.Instance;
            if (_sampleReadback || volume == null || Baked == 0) return false;
            BindVolume(volume);
            _bake.SetVector(SamplePointId, world);
            _bake.Dispatch(_sampleKernel, 1, 1, 1);
            _sampleReadback = true;
            int clears = _clears;
            AsyncGPUReadback.Request(_sample, request =>
            {
                _sampleReadback = false;
                if (request.hasError || _sample == null) return;
                var data = request.GetData<float>();
                done(clears == _clears && data[1] > 0.01f ? data[0] : null);
            });
            return true;
        }

        void Update()
        {
            if (Baked == 0 || _histogramReadback || _useReadback || Time.unscaledTime < _nextScene) return;
            _nextScene = Time.unscaledTime + SceneInterval;
            _bake.Dispatch(_clearHistogramKernel, (HistogramBins + 63) / 64, 1, 1);
            _bake.Dispatch(_histogramKernel, MapSize / 2 / 8, MapSize / 2 / 8, 1);
            _histogramReadback = _useReadback = true;
            int clears = _clears;
            int stamp = _keyframes.BeginCount();
            AsyncGPUReadback.Request(_keyframeUse, request =>
            {
                _useReadback = false;
                if (request.hasError || clears != _clears || _keyframeUse == null) return;
                _keyframes.SetUse(request.GetData<uint>(), stamp);
            });
            AsyncGPUReadback.Request(_histogram, request =>
            {
                _histogramReadback = false;
                // A Clear since the request makes this answer stale.
                if (request.hasError || clears != _clears || _histogram == null) return;
                SceneRange = Percentiles(request.GetData<uint>());
            });
        }

        static Vector2? Percentiles(Unity.Collections.NativeArray<uint> histogram)
        {
            long total = 0;
            foreach (uint count in histogram) total += count;
            if (total == 0) return null;
            float lo = float.NaN, hi = float.NaN;
            long sum = 0;
            for (int i = 0; i < histogram.Length; i++)
            {
                sum += histogram[i];
                if (float.IsNaN(lo) && sum >= SceneLow * total) lo = HistogramMin + i * HistogramStep;
                if (sum >= SceneHigh * total) { hi = HistogramMin + (i + 1) * HistogramStep; break; }
            }
            return new Vector2(lo, hi);
        }

        void LateUpdate()
        {
            if (!_visible) return;
            _style.Apply(_material);
            _keyframes.Bind(_material);
            _alignment.Lens.Apply(_material);
            _panel.FlatField.Apply(_material);
            _material.SetFloat(NowId, Time.unscaledTime);
            _material.SetTexture(PaletteId, _panel.Palette);
            _material.SetFloat(MinId, _panel.RangeMin);
            _material.SetFloat(MaxId, _panel.RangeMax);
        }

        void OnDestroy()
        {
            if (_panel != null && _panel.Thermal != null) _panel.Thermal.FrameReady -= OnThermal;
            GPUMeshRenderer.OverlayMaterials.Remove(_material);
            Destroy(_material);
            if (_xz != null) _xz.Release();
            if (_xy != null) _xy.Release();
            if (_yz != null) _yz.Release();
            if (_tagXz != null) _tagXz.Release();
            if (_tagXy != null) _tagXy.Release();
            if (_tagYz != null) _tagYz.Release();
            _keyframes?.Dispose();
            _histogram?.Release();
            _histogram = null;
            _sample?.Release();
            _sample = null;
            _keyframeUse?.Release();
            _keyframeUse = null;
        }
    }
}
