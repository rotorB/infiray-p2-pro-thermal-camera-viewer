// Rounded rectangle for the panel and the HUD: a quad of _Size metres, corners of _Radius,
// an optional _BorderWidth outline, edges antialiased from the distance field. _TexColour = 1
// fills it from _MainTex instead (the palette bar). Set per renderer through a
// MaterialPropertyBlock (UiRect). _ZTest = Always for the HUD.
// Lives in Resources so the build includes it although only code creates its material.
Shader "P2Pro/UIRect"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Colour ("Colour", Color) = (0.1, 0.1, 0.12, 0.9)
        _BorderColour ("Border colour", Color) = (1, 1, 1, 0.15)
        _Size ("Size (m)", Vector) = (0.1, 0.05, 0, 0)
        _Radius ("Corner radius (m)", Float) = 0.006
        _BorderWidth ("Border width (m)", Float) = 0
        _TexColour ("Colour from texture", Float) = 0
        [Enum(UnityEngine.Rendering.CompareFunction)] _ZTest ("ZTest", Float) = 4
    }
    SubShader
    {
        Tags { "RenderType" = "Transparent" "RenderPipeline" = "UniversalPipeline" "Queue" = "Transparent+99" }
        Pass
        {
            Cull Off
            ZWrite Off
            ZTest [_ZTest]
            Blend SrcAlpha OneMinusSrcAlpha, One OneMinusSrcAlpha
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_MainTex); SAMPLER(sampler_MainTex);
            CBUFFER_START(UnityPerMaterial)
                float4 _MainTex_ST;
                half4 _Colour;
                half4 _BorderColour;
                float4 _Size;
                float _Radius;
                float _BorderWidth;
                half _TexColour;
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
                o.uv = v.uv;
                return o;
            }

            half4 frag(Varyings i) : SV_Target
            {
                float2 half_ = _Size.xy * 0.5;
                float r = min(_Radius, min(half_.x, half_.y));
                float2 p = (i.uv - 0.5) * _Size.xy;
                float2 q = abs(p) - half_ + r;
                float d = length(max(q, 0)) + min(max(q.x, q.y), 0) - r;     // < 0 inside
                float aa = max(fwidth(d), 1e-5);
                float coverage = saturate(0.5 - d / aa);
                if (coverage <= 0) discard;
                half4 fill = _TexColour > 0.5 ? half4(SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv).rgb, _Colour.a) : _Colour;
                half border = _BorderWidth > 0 ? saturate(0.5 + (d + _BorderWidth) / aa) : 0;
                half4 c = lerp(fill, _BorderColour, border);
                c.a *= coverage;
                return c;
            }
            ENDHLSL
        }
    }
}
