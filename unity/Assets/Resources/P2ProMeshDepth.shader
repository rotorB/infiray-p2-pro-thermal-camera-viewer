// Depth only of QuestRoomScan's GPU mesh, drawn first through GPUMeshRenderer.OverlayMaterials
// while the scan's own mesh is hidden: the thermal overlays (ZTest LEqual) then show only the
// nearest surface, not the walls behind it. Writes no colour, so passthrough stays as it is.
// Lives in Resources so the build includes it although only code creates its material.
Shader "P2Pro/MeshDepth"
{
    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "Queue" = "Geometry" }
        Pass
        {
            Tags { "LightMode" = "SRPDefaultUnlit" }
            Cull Off
            ZWrite On
            ZTest LEqual
            ColorMask 0
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

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

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(uint vertID : SV_VertexID)
            {
                Varyings o = (Varyings)0;
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.positionHCS = TransformWorldToHClip(_SurfaceVerts[_SurfaceIndices[vertID]].pos);
                return o;
            }

            half4 frag(Varyings i) : SV_Target { return 0; }
            ENDHLSL
        }
    }
}
