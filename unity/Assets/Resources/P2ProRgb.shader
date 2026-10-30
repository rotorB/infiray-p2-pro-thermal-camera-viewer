// Plain textured quad for the headset RGB camera image in the debug view.
// Lives in Resources so the build includes it although only code creates its material.
Shader "P2Pro/Rgb"
{
    Properties
    {
        _MainTex ("Image", 2D) = "black" {}
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "Queue" = "Geometry" }
        Pass
        {
            Cull Off
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_MainTex); SAMPLER(sampler_MainTex);
            TEXTURE2D(_EdgeTex); SAMPLER(sampler_EdgeTex);
            CBUFFER_START(UnityPerMaterial)
                float4 _MainTex_ST;
                float _PanelEdges;      // 1: the RGB edge map (Edge align) drawn cyan over the image
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
            };

            Varyings vert(Attributes v)
            {
                Varyings o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.positionCS = TransformObjectToHClip(v.positionOS.xyz);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            half4 frag(Varyings i) : SV_Target
            {
                half3 c = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv).rgb;
                if (_PanelEdges > 0.5) c = lerp(c * 0.6, half3(0, 1, 1), SAMPLE_TEXTURE2D(_EdgeTex, sampler_EdgeTex, i.uv).r);
                return half4(c, 1);
            }
            ENDHLSL
        }
    }
}
