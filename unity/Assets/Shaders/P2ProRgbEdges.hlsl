// Edges of the headset RGB camera image at the thermal camera's pixel size (RgbEdgeMap,
// P2ProRgbEdgeMap.compute): drawn in cyan in the Edges look and collected by
// P2ProEdgeAlign.compute. _RgbEdgeTex covers the RGB image's texture coordinates.
#ifndef P2PRO_RGB_EDGES_INCLUDED
#define P2PRO_RGB_EDGES_INCLUDED

Texture2D<float> _RgbEdgeTex;
SamplerState sampler_linear_clamp;

// 0..1 on an RGB edge, 0 elsewhere or outside the image.
float RgbEdgeStrength(float2 uv)
{
    if (any(uv < 0) || any(uv > 1)) return 0;
    return _RgbEdgeTex.SampleLevel(sampler_linear_clamp, uv, 0);
}

#endif
