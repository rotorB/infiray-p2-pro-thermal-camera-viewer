// Final colour of a thermal overlay on the room mesh, shared by P2Pro/ThermalLive and
// P2Pro/ThermalMesh. Set by OverlayStyle, together with the pass's blend factors
// (_SrcBlend, _DstBlend, _SrcBlendA, _DstBlendA).
//
// Passthrough is composited under the app as rgb + (1 - alpha) * passthrough, so:
//   alpha    (palette, a)            -> palette over the room at opacity a
//   add      (palette * a, 0)        -> palette added to the room
//   screen   (palette * a, lum * a)  -> screen, by the palette's brightness
//   multiply (0, (1 - lum) * a)      -> room darkened where the palette is dark
// Filter: 0 = all, 1 = only above _Threshold °C, 2 = only below.
// _Wire = 1 draws only the triangle edges, hair-thin and antialiased (each edge coloured by
// its temperature): the coverage scales the opacity.
#ifndef P2PRO_OVERLAY_INCLUDED
#define P2PRO_OVERLAY_INCLUDED

float _Alpha;
float _Mode;
float _Filter;
float _Threshold;
float _Wire;

static const float WireWidthPx = 0.25;   // core; plus a 1 px soft edge

// Barycentric corner for the mesh's non-indexed draw: vertex vertID % 3 of its triangle.
float3 WireBarycentric(uint vertID)
{
    uint corner = vertID % 3;
    return corner == 0 ? float3(1, 0, 0) : corner == 1 ? float3(0, 1, 0) : float3(0, 0, 1);
}

// 1 outside wireframe mode; in it, how much of the pixel the nearest edge covers (fragments
// off every edge are dropped).
float WireCoverage(float3 bary)
{
    if (_Wire < 0.5) return 1;
    float3 px = bary / max(fwidth(bary), 1e-6);
    float coverage = saturate(WireWidthPx + 0.5 - min(px.x, min(px.y, px.z)));
    if (coverage <= 0) discard;
    return coverage;
}

half4 OverlayColour(half3 palette, float celsius, float coverage = 1)
{
    if ((_Filter > 0.5 && _Filter < 1.5 && celsius < _Threshold) || (_Filter > 1.5 && celsius > _Threshold))
        discard;
    half lum = dot(palette, half3(0.299, 0.587, 0.114));
    half a = _Alpha * coverage;
    half4 c;
    if (_Mode < 0.5) c = half4(palette, a);
    else if (_Mode < 1.5) c = half4(palette * a, 0);
    else if (_Mode < 2.5) c = half4(palette * a, lum * a);
    else c = half4(0, 0, 0, (1 - lum) * a);
    return c;
}

#endif
