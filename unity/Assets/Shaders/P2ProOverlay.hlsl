// Final colour of a thermal overlay on the room mesh, shared by P2Pro/ThermalLive,
// P2Pro/ThermalMesh and P2Pro/ThermalFlat. Set by OverlayStyle, together with the pass's blend factors
// (_SrcBlend, _DstBlend, _SrcBlendA, _DstBlendA).
//
// Passthrough is composited under the app as rgb + (1 - alpha) * passthrough, so:
//   alpha    (palette, a)            -> palette over the room at opacity a
//   add      (palette * a, 0)        -> palette added to the room
//   screen   (palette * a, lum * a)  -> screen, by the palette's brightness
//   multiply (0, (1 - lum) * a)      -> room darkened where the palette is dark
// Filter: 0 = all, 1 = only inside the band _Threshold .. _ThresholdHigh °C, 2 = only outside it.
// _Wire = 1 draws only the triangle edges, hair-thin and antialiased (each edge coloured by
// its temperature): the coverage scales the opacity.
// _Edges = 1 draws only edges (P2ProThermalEdges.hlsl): the thermal image's in magenta and,
// where the shader has it, the headset RGB camera's in cyan; where both meet they add to white.
// Flashlight (ThermalFlashlight, globals): with _FlashOn the overlay shows only inside a cone
// from the right controller, fading out over its rim; elsewhere the room stays as it is.
#ifndef P2PRO_OVERLAY_INCLUDED
#define P2PRO_OVERLAY_INCLUDED

float _Alpha;
float _Mode;
float _Filter;
float _Threshold;
float _ThresholdHigh;
float _Wire;
float _Edges;
float _EdgeThreshold;   // °C per thermal camera pixel
float _FlashOn;
float3 _FlashPos;
float3 _FlashDir;
float2 _FlashCos;       // cos of the cone's outer and inner half-angles

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

// 1 inside the flashlight's cone (or with the flashlight off), 0 outside it.
float FlashMask(float3 world)
{
    if (_FlashOn < 0.5) return 1;
    float c = dot(normalize(world - _FlashPos), _FlashDir);
    return smoothstep(_FlashCos.x, _FlashCos.y, c);
}

// colour at opacity a in the current blend mode.
half4 BlendColour(half3 palette, half a)
{
    half lum = dot(palette, half3(0.299, 0.587, 0.114));
    half4 c;
    if (_Mode < 0.5) c = half4(palette, a);
    else if (_Mode < 1.5) c = half4(palette * a, 0);
    else if (_Mode < 2.5) c = half4(palette * a, lum * a);
    else c = half4(0, 0, 0, (1 - lum) * a);
    return c;
}

half4 OverlayColour(half3 palette, float celsius, float coverage, float3 world)
{
    bool inside = celsius >= _Threshold && celsius <= _ThresholdHigh;
    if ((_Filter > 0.5 && _Filter < 1.5 && !inside) || (_Filter > 1.5 && inside))
        discard;
    float flash = FlashMask(world);
    if (flash <= 0) discard;
    return BlendColour(palette, _Alpha * coverage * flash);
}

// The Edges look: magenta where the thermal edge is, cyan where the RGB edge is, white where
// both are, nothing elsewhere; opaque, so thin lines stay visible.
half4 EdgeColour(float thermalEdge, float rgbEdge, float3 world)
{
    float a = max(thermalEdge, rgbEdge) * FlashMask(world);
    if (a <= 0) discard;
    half3 c = saturate(thermalEdge * half3(1, 0, 1) + rgbEdge * half3(0, 1, 1));
    return BlendColour(c / max(max(c.r, c.g), max(c.b, 1e-3)), a);
}

#endif
