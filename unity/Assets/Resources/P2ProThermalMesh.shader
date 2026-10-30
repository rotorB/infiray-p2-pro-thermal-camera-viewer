// Draws the persistent thermal map (P2ProThermalBake.compute) on QuestRoomScan's GPU mesh, as an
// extra pass through GPUMeshRenderer.OverlayMaterials. The maps hold °C; the palette is applied
// here, last. Surfaces without thermal data stay untouched.
// Detail: each map texel names up to four keyframes that saw it (ThermalKeyframes); the fragment
// is projected into each of those whole stored frames like P2Pro/ThermalLive does, and their °C
// are averaged, weighted by how near each image's centre the point lies and by age. The room
// keeps the thermal camera's resolution, and views fade into each other instead of meeting at
// seams. Without a usable keyframe the texel's own °C is shown. Only what lands on the room is
// averaged; the live image stays as the camera sends it.
// Lives in Resources so the build includes it although only code creates its material.
Shader "P2Pro/ThermalMesh"
{
    Properties
    {
        _Palette ("Palette", 2D) = "white" {}
        _Min ("Range min (°C)", Float) = 15
        _Max ("Range max (°C)", Float) = 35
        _Alpha ("Opacity", Range(0, 1)) = 0.75
        _Mode ("Blend mode", Float) = 0
        _Filter ("Filter", Float) = 0
        _Threshold ("Threshold (°C)", Float) = 30
        _Wire ("Wireframe", Float) = 0
        [Enum(UnityEngine.Rendering.BlendMode)] _SrcBlend ("Src", Float) = 5
        [Enum(UnityEngine.Rendering.BlendMode)] _DstBlend ("Dst", Float) = 10
        [Enum(UnityEngine.Rendering.BlendMode)] _SrcBlendA ("Src alpha", Float) = 1
        [Enum(UnityEngine.Rendering.BlendMode)] _DstBlendA ("Dst alpha", Float) = 10
    }
    SubShader
    {
        Tags { "RenderType" = "Transparent" "RenderPipeline" = "UniversalPipeline" "Queue" = "Transparent" }
        Pass
        {
            Tags { "LightMode" = "SRPDefaultUnlit" }
            Cull Off
            ZWrite Off
            ZTest LEqual
            Offset -1, -1
            Blend [_SrcBlend] [_DstBlend], [_SrcBlendA] [_DstBlendA]
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Assets/Shaders/P2ProOverlay.hlsl"
            #include "Assets/Shaders/P2ProThermalLens.hlsl"
            #include "Assets/Shaders/P2ProFlatOffset.hlsl"

            struct GPUVertex
            {
                float3 pos;
                float3 prevPos;
                float3 norm;
                uint packedColor;
                uint voxelFlatIdx;
                uint _pad;
            };
            StructuredBuffer<GPUVertex> _SurfaceVerts;
            StructuredBuffer<uint> _SurfaceIndices;

            Texture2D<float4> _MapXZ;
            Texture2D<float4> _MapXY;
            Texture2D<float4> _MapYZ;
            Texture2D<float4> _TagXZ;
            Texture2D<float4> _TagXY;
            Texture2D<float4> _TagYZ;
            SamplerState sampler_point_clamp;
            TEXTURE2D(_Palette); SAMPLER(sampler_Palette);
            float _Min;
            float _Max;
            float4 gsVoxCount;      // set globally by QuestRoomScan
            float gsVoxSize;

            // As ThermalKeyframes.Params.
            struct Keyframe
            {
                float4 row0, row1, row2;    // world -> RGB camera
                float4 position;            // RGB camera; w = tag
                float4 intrinsics;
                float4 crop;
                float4 mapA;
                float4 mapB;
                float4 info;                // x = time taken
            };
            #define KEYFRAMES 128
            StructuredBuffer<Keyframe> _Keyframes;
            Texture2DArray<float> _KeyframeTex;
            float _Now;
            // Seconds over which an old keyframe's say halves against newer ones.
            #define KEYFRAME_HALF_LIFE 60

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 bary : TEXCOORD5;
                float3 normalWS : TEXCOORD1;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings o = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                GPUVertex v = _SurfaceVerts[_SurfaceIndices[vertID]];
                o.positionWS = v.pos;
                o.normalWS = v.norm;
                o.positionHCS = TransformWorldToHClip(v.pos);
                o.bary = WireBarycentric(vertID);
                return o;
            }

            // Adds one map's temperature, weighted, if it has data for this surface; the keyframes
            // of the most weighted one are kept.
            void Accumulate(Texture2D<float4> map, Texture2D<float4> tagMap, float2 baseUv, float normalSign,
                            float along, float weight, inout float sum, inout float weights, inout float best,
                            inout float4 tags)
            {
                float2 uv = float2(baseUv.x, baseUv.y * 0.5 + (normalSign > 0 ? 0.5 : 0));
                float4 t = map.SampleLevel(sampler_point_clamp, uv, 0);
                if (t.y > 0.01 && abs(t.z - along) < 0.02)
                {
                    sum += t.x * weight;
                    weights += weight;
                    if (weight > best) { best = weight; tags = tagMap.SampleLevel(sampler_point_clamp, uv, 0); }
                }
            }

            // °C of a world point in a stored keyframe (bilinear by hand, less the sensor offsets),
            // and its weight: falls to 0 at the image edge, halves every KEYFRAME_HALF_LIFE.
            // False when the keyframe was overwritten or the point falls outside its image.
            // Same projection as ThermalLive.
            bool KeyframeCelsius(float3 world, float tag, out float celsius, out float weight)
            {
                celsius = 0;
                weight = 0;
                if (tag < 0.5) return false;
                uint slot = ((uint)round(tag) - 1) % KEYFRAMES;
                Keyframe k = _Keyframes[slot];
                if (abs(k.position.w - tag) > 0.25) return false;
                float4 p = float4(world, 1);
                float3 c = float3(dot(k.row0, p), dot(k.row1, p), dot(k.row2, p));
                float2 uv;
                if (!ThermalUvOf(c, k.intrinsics, k.crop, k.mapA, k.mapB, uv) || any(uv < 0) || any(uv > 1)) return false;

                uint width, height, slices;
                _KeyframeTex.GetDimensions(width, height, slices);
                int2 size = int2(width, height);
                float2 f2 = uv * (float2)size - 0.5;
                int2 i = (int2)floor(f2);
                float2 f = f2 - i;
                int2 i0 = clamp(i, 0, size - 1);
                int2 i1 = clamp(i + 1, 0, size - 1);
                float bottom = lerp(_KeyframeTex.Load(int4(i0.x, i0.y, slot, 0)), _KeyframeTex.Load(int4(i1.x, i0.y, slot, 0)), f.x);
                float top = lerp(_KeyframeTex.Load(int4(i0.x, i1.y, slot, 0)), _KeyframeTex.Load(int4(i1.x, i1.y, slot, 0)), f.x);
                celsius = lerp(bottom, top, f.y) - FlatOffset(uv);
                float2 d = abs(uv - 0.5) * 2;
                float centrality = 1 - max(d.x, d.y);
                weight = centrality * centrality * exp2(-(_Now - k.info.x) / KEYFRAME_HALF_LIFE);
                return true;
            }

            half4 frag(Varyings i) : SV_Target
            {
                float coverage = WireCoverage(i.bary);
                float3 uvw = i.positionWS / (gsVoxCount.xyz * gsVoxSize) + 0.5;
                float3 n = normalize(i.normalWS);
                float3 w = abs(n) / (abs(n.x) + abs(n.y) + abs(n.z) + 1e-4);
                float sum = 0, weights = 0, best = 0;
                float4 tags = 0;
                Accumulate(_MapXZ, _TagXZ, uvw.xz, n.y, uvw.y, w.y, sum, weights, best, tags);
                Accumulate(_MapXY, _TagXY, uvw.xy, n.z, uvw.z, w.z, sum, weights, best, tags);
                Accumulate(_MapYZ, _TagYZ, uvw.yz, n.x, uvw.x, w.x, sum, weights, best, tags);
                if (weights < 1e-3) discard;
                float celsius = sum / weights;
                float frameSum = 0, frameWeights = 0;
                [unroll] for (int s = 0; s < 4; s++)
                {
                    float c, weight;
                    if (KeyframeCelsius(i.positionWS, tags[s], c, weight)) { frameSum += c * weight; frameWeights += weight; }
                }
                if (frameWeights > 1e-5) celsius = frameSum / frameWeights;
                float k = saturate((celsius - _Min) / max(_Max - _Min, 0.01));
                return OverlayColour(SAMPLE_TEXTURE2D(_Palette, sampler_Palette, float2(k, 0.5)).rgb, celsius, coverage);
            }
            ENDHLSL
        }
    }
}
