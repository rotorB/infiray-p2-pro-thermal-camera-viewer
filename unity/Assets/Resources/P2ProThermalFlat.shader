// The thermal image as a flat quad in front of the RGB camera (ThermalFlat): no projection on
// the room, like the overlay in calibration. °C -> palette here, last; blend, opacity and
// filter from OverlayStyle (P2ProOverlay.hlsl).
// Lives in Resources so the build includes it although only code creates its material.
Shader "P2Pro/ThermalFlat"
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
        [Enum(UnityEngine.Rendering.BlendMode)] _SrcBlend ("Src", Float) = 5
        [Enum(UnityEngine.Rendering.BlendMode)] _DstBlend ("Dst", Float) = 10
        [Enum(UnityEngine.Rendering.BlendMode)] _SrcBlendA ("Src alpha", Float) = 1
        [Enum(UnityEngine.Rendering.BlendMode)] _DstBlendA ("Dst alpha", Float) = 10
    }
    SubShader
    {
        Tags { "RenderType" = "Transparent" "RenderPipeline" = "UniversalPipeline" "Queue" = "Transparent+10" }
        Pass
        {
            Cull Off
            ZWrite Off
            ZTest Always
            Blend [_SrcBlend] [_DstBlend], [_SrcBlendA] [_DstBlendA]
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Assets/Shaders/P2ProOverlay.hlsl"

            Texture2D<float> _MainTex;
            TEXTURE2D(_Palette); SAMPLER(sampler_Palette);
            float _Min;
            float _Max;

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
                o.uv = v.uv;
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

            half4 frag(Varyings i) : SV_Target
            {
                float celsius = Celsius(i.uv);
                float k = saturate((celsius - _Min) / max(_Max - _Min, 0.01));
                return OverlayColour(SAMPLE_TEXTURE2D(_Palette, sampler_Palette, float2(k, 0.5)).rgb, celsius);
            }
            ENDHLSL
        }
    }
}
