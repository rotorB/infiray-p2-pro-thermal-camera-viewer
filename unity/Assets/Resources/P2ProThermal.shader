// Maps a temperature texture (°C, single channel) through a 256x1 palette between _Min and _Max.
// Lives in Resources so the build includes it although only code creates its material.
Shader "P2Pro/Thermal"
{
    Properties
    {
        _MainTex ("Temperature (°C)", 2D) = "black" {}
        _Palette ("Palette", 2D) = "white" {}
        _Min ("Range min (°C)", Float) = 15
        _Max ("Range max (°C)", Float) = 35
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
            TEXTURE2D(_Palette); SAMPLER(sampler_Palette);
            CBUFFER_START(UnityPerMaterial)
                float4 _MainTex_ST;
                float _Min;
                float _Max;
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
                float celsius = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv).r;
                float t = saturate((celsius - _Min) / max(_Max - _Min, 0.01));
                return half4(SAMPLE_TEXTURE2D(_Palette, sampler_Palette, float2(t, 0.5)).rgb, 1);
            }
            ENDHLSL
        }
    }
}
