// Projects the latest thermal frame onto QuestRoomScan's GPU mesh (drawn as an extra pass by
// GPUMeshRenderer.OverlayMaterials), without keeping anything. Per fragment: world point ->
// headset RGB camera at the frame's exposure -> thermal texture coordinate through the lens
// model or the Align-mode mapping (P2ProThermalLens.hlsl) -> °C -> palette (see RgbProjection).
// Lives in Resources so the build includes it although only code creates its material.
Shader "P2Pro/ThermalLive"
{
    Properties
    {
        _MainTex ("Temperature (°C)", 2D) = "black" {}
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

            Texture2D<float> _MainTex;
            TEXTURE2D(_Palette); SAMPLER(sampler_Palette);
            float _Min;
            float _Max;
            float4x4 _WorldToRgb;
            float3 _RgbPosition;
            float4 _RgbIntrinsics;
            float4 _RgbCrop;
            float4 _MapA;
            float4 _MapB;
            float4x4 _WorldToRgbImage;          // world -> RGB camera when the RGB image was taken
            float _RgbEdgesOn;

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 bary : TEXCOORD5;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings o = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                GPUVertex v = _SurfaceVerts[_SurfaceIndices[vertID]];
                o.positionWS = v.pos;
                o.positionHCS = TransformWorldToHClip(v.pos);
                o.bary = WireBarycentric(vertID);
                return o;
            }

            // Bilinear by hand: linear filtering of 32-bit float textures is optional on mobile GPUs.
            float Celsius(float2 uv)
            {
                uint w, h;
                _MainTex.GetDimensions(w, h);       // 256 x 192, or 3x that when sharpened
                int2 size = int2(w, h);
                float2 p = uv * (float2)size - 0.5;
                int2 i = (int2)floor(p);
                float2 f = p - i;
                int2 i0 = clamp(i, 0, size - 1);
                int2 i1 = clamp(i + 1, 0, size - 1);
                float bottom = lerp(_MainTex.Load(int3(i0.x, i0.y, 0)), _MainTex.Load(int3(i1.x, i0.y, 0)), f.x);
                float top = lerp(_MainTex.Load(int3(i0.x, i1.y, 0)), _MainTex.Load(int3(i1.x, i1.y, 0)), f.x);
                return lerp(bottom, top, f.y);
            }
            #include "Assets/Shaders/P2ProThermalEdges.hlsl"
            #include "Assets/Shaders/P2ProRgbEdges.hlsl"

            // The RGB image's texture coordinate of a world point.
            float2 RgbImageUv(float3 world)
            {
                float3 c = mul(_WorldToRgbImage, float4(world, 1)).xyz;
                if (c.z <= 0.05) return float2(-1, -1);
                float2 sensor = c.xy / c.z * _RgbIntrinsics.xy + _RgbIntrinsics.zw;
                return (sensor - _RgbCrop.xy) / _RgbCrop.zw;
            }

            half4 frag(Varyings i) : SV_Target
            {
                float coverage = WireCoverage(i.bary);
                float3 c = mul(_WorldToRgb, float4(i.positionWS, 1)).xyz;
                float2 uv;
                if (!ThermalUvOf(c, _RgbIntrinsics, _RgbCrop, _MapA, _MapB, uv) || any(uv < 0) || any(uv > 1)) discard;
                if (_Edges > 0.5)
                    return EdgeColour(EdgeStrength(uv), _RgbEdgesOn > 0.5 ? RgbEdgeStrength(RgbImageUv(i.positionWS)) : 0, i.positionWS);
                float celsius = Celsius(uv);
                float k = saturate((celsius - _Min) / max(_Max - _Min, 0.01));
                return OverlayColour(SAMPLE_TEXTURE2D(_Palette, sampler_Palette, float2(k, 0.5)).rgb, celsius, coverage, i.positionWS);
            }
            ENDHLSL
        }
    }
}
