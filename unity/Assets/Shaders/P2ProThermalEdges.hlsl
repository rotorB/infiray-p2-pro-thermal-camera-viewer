// Edges look (OverlayStyle): the thermal image's edges as 1-camera-pixel lines, to check the
// camera alignment against the real outlines seen through passthrough, also at the image's
// edges and corners. Include after the shader's own `float Celsius(float2 uv)`.
#ifndef P2PRO_THERMAL_EDGES_INCLUDED
#define P2PRO_THERMAL_EDGES_INCLUDED

static const float2 EdgePixel = 1.0 / float2(256, 192);    // one thermal camera pixel in uv

// °C per camera pixel along x and y (central differences).
float2 ThermalGradient(float2 uv)
{
    return 0.5 * float2(Celsius(uv + float2(EdgePixel.x, 0)) - Celsius(uv - float2(EdgePixel.x, 0)),
                        Celsius(uv + float2(0, EdgePixel.y)) - Celsius(uv - float2(0, EdgePixel.y)));
}

// 0..1 on the ridge of the gradient (non-maximum suppression one pixel either side across the
// edge), where it is steeper than _EdgeThreshold °C per pixel; 0 elsewhere.
float EdgeStrength(float2 uv)
{
    float2 g = ThermalGradient(uv);
    float m = length(g);
    if (m < _EdgeThreshold) return 0;
    float2 across = g / m * EdgePixel;
    if (m < length(ThermalGradient(uv + across)) || m < length(ThermalGradient(uv - across))) return 0;
    return smoothstep(_EdgeThreshold, 1.5 * _EdgeThreshold, m);
}


#endif
