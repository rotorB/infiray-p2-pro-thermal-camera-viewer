using System;
using Genesis.RoomScan;
using UnityEngine;
using UnityEngine.Rendering;

namespace P2Pro
{
    /// <summary>
    /// Measures how far world rays travel before they hit the room, from QuestRoomScan's live
    /// environment depth, on the GPU. One request in flight at a time; the answer arrives a frame
    /// or two later.
    /// </summary>
    public class DepthProbe : MonoBehaviour
    {
        const int MaxRays = 16;

        static readonly int DepthTexId = Shader.PropertyToID("gsDepthTex");
        static readonly int ProjId = Shader.PropertyToID("gsDepthProj");
        static readonly int ProjInvId = Shader.PropertyToID("gsDepthProjInv");
        static readonly int ViewId = Shader.PropertyToID("gsDepthView");
        static readonly int ViewInvId = Shader.PropertyToID("gsDepthViewInv");
        static readonly int ZParamsId = Shader.PropertyToID("gsDepthZParams");
        static readonly int RaysId = Shader.PropertyToID("_Rays");
        static readonly int ResultId = Shader.PropertyToID("_Result");
        static readonly int RayCountId = Shader.PropertyToID("_RayCount");

        ComputeShader _shader;
        int _kernel;
        ComputeBuffer _rays;
        ComputeBuffer _result;
        readonly Vector4[] _rayData = new Vector4[MaxRays * 2];
        bool _busy;

        void Awake()
        {
            _shader = Resources.Load<ComputeShader>("P2ProDepthProbe");
            _kernel = _shader.FindKernel("Probe");
            _rays = new ComputeBuffer(MaxRays * 2, 16);
            _result = new ComputeBuffer(MaxRays, 4);
        }

        /// <summary>
        /// Starts measuring world-space <paramref name="rays"/>; <paramref name="done"/> gets one
        /// distance per ray in metres, negative where nothing was hit. False if busy or no depth.
        /// </summary>
        public bool Request(Ray[] rays, Action<float[]> done)
        {
            var depth = DepthCapture.Instance;
            if (_busy || depth == null || !DepthCapture.DepthAvailable || depth.DepthTex == null) return false;
            int n = Mathf.Min(rays.Length, MaxRays);
            for (int i = 0; i < n; i++)
            {
                _rayData[i * 2] = rays[i].origin;
                _rayData[i * 2 + 1] = rays[i].direction;
            }
            _rays.SetData(_rayData);
            _shader.SetTexture(_kernel, DepthTexId, depth.DepthTex);
            _shader.SetMatrixArray(ProjId, depth.Proj);
            _shader.SetMatrixArray(ProjInvId, depth.ProjInv);
            _shader.SetMatrixArray(ViewId, depth.View);
            _shader.SetMatrixArray(ViewInvId, depth.ViewInv);
            _shader.SetVector(ZParamsId, depth.Planes);
            _shader.SetInt(RayCountId, n);
            _shader.SetBuffer(_kernel, RaysId, _rays);
            _shader.SetBuffer(_kernel, ResultId, _result);
            _shader.Dispatch(_kernel, (n + 7) / 8, 1, 1);

            _busy = true;
            AsyncGPUReadback.Request(_result, n * 4, 0, request =>
            {
                _busy = false;
                if (request.hasError) return;
                done(request.GetData<float>().ToArray());
            });
            return true;
        }

        /// <summary>Median of the non-negative distances, or -1.</summary>
        public static float Median(float[] distances)
        {
            var hits = Array.FindAll(distances, d => d > 0);
            if (hits.Length == 0) return -1;
            Array.Sort(hits);
            return hits[hits.Length / 2];
        }

        void OnDestroy()
        {
            _rays?.Release();
            _result?.Release();
        }
    }
}
