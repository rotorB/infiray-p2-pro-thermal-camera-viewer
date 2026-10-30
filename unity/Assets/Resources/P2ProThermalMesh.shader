// Draws the persistent thermal map (P2ProThermalBake.compute) on QuestRoomScan's GPU mesh, as an
// extra pass through GPUMeshRenderer.OverlayMaterials. The maps hold °C; the palette is applied
// here, last. Surfaces without thermal data stay untouched. How a point gets its °C:
// P2ProThermalPaint.hlsl.
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
            #include "Assets/Shaders/P2ProThermalPaint.hlsl"

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

            TEXTURE2D(_Palette); SAMPLER(sampler_Palette);
            float _Min;
            float _Max;

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

            half4 frag(Varyings i) : SV_Target
            {
                float coverage = WireCoverage(i.bary);
                float celsius;
                if (!PaintCelsius(i.positionWS, i.normalWS, celsius)) discard;
                float k = saturate((celsius - _Min) / max(_Max - _Min, 0.01));
                return OverlayColour(SAMPLE_TEXTURE2D(_Palette, sampler_Palette, float2(k, 0.5)).rgb, celsius, coverage, i.positionWS);
            }
            ENDHLSL
        }
    }
}
