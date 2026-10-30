// World -> thermal image, shared by P2ProThermalBake.compute, P2Pro/ThermalMesh and
// P2Pro/ThermalLive. A point already in the headset RGB camera's frame (c, metres) goes to the
// thermal texture through the fitted camera model (ThermalLens: rotation, position, focal
// lengths, principal point, radial k1 k2) when it is solved, else through the Align-mode
// mapping A + B / distance on the RGB image (ThermalAlignment). Uniforms from
// ThermalLens.Apply and RgbProjection.Apply.
#ifndef P2PRO_THERMAL_LENS_INCLUDED
#define P2PRO_THERMAL_LENS_INCLUDED

float _LensOn;
float4 _LensRow0, _LensRow1, _LensRow2; // RGB camera frame -> thermal camera frame (w = offset)
float4 _LensK;                          // fx, fy, cx, cy in thermal pixels, y up
float4 _LensD;                          // k1, k2, image width, height in pixels
float _RgbAspect;
float _ThermalAspect;

// Thermal texture coordinate of c; false behind either camera. The result may lie outside 0..1.
// intrinsics (fx, fy, cx, cy) and crop (image rectangle) are the RGB camera's sensor, mapA and
// mapB the mapping, both for the legacy path.
bool ThermalUvOf(float3 c, float4 intrinsics, float4 crop, float4 mapA, float4 mapB, out float2 uv)
{
    uv = float2(-1, -1);
    if (c.z <= 0.05) return false;
    if (_LensOn > 0.5)
    {
        float4 p = float4(c, 1);
        float3 t = float3(dot(_LensRow0, p), dot(_LensRow1, p), dot(_LensRow2, p));
        if (t.z <= 0.05) return false;
        float2 x = t.xy / t.z;
        float r2 = dot(x, x);
        if (r2 > 1.5) return false;     // far outside the field: the distortion polynomial folds
        uv = (_LensK.zw + _LensK.xy * x * (1 + r2 * (_LensD.x + r2 * _LensD.y))) / _LensD.zw;
        return true;
    }
    float2 sensor = c.xy / c.z * intrinsics.xy + intrinsics.zw;
    float2 rgb = (sensor - crop.xy) / crop.zw;
    float4 m = mapA + mapB / max(length(c), 0.1);
    float2 w = float2(rgb.x - m.x, (rgb.y - m.y) * _RgbAspect);
    float s, co;
    sincos(-m.w, s, co);
    float2 t = float2(co * w.x - s * w.y, s * w.x + co * w.y) / m.z;
    uv = float2(t.x + 0.5, t.y / _ThermalAspect + 0.5);
    return true;
}

#endif
