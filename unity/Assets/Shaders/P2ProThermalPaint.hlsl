// °C of a room surface from the persistent thermal map (P2ProThermalBake.compute), for
// P2Pro/ThermalMesh. Needs P2ProThermalLens.hlsl and P2ProFlatOffset.hlsl included first.
// Each map texel names up to four keyframes that saw it (ThermalKeyframes); the point is
// projected into each of those whole stored frames like P2Pro/ThermalLive does, and their °C are
// averaged, weighted by how near each image's centre the point lies and by age. The room keeps
// the thermal camera's resolution, and views fade into each other instead of meeting at seams.
// Without a usable keyframe the texel's own °C is used.
// A rebuilt mesh triangle may sit a little off where its surface was baked, or have turned: when
// none of the maps has data right under it, texels a few centimetres around are used instead.
#ifndef P2PRO_THERMAL_PAINT_INCLUDED
#define P2PRO_THERMAL_PAINT_INCLUDED

Texture2D<float4> _MapXZ;
Texture2D<float4> _MapXY;
Texture2D<float4> _MapYZ;
Texture2D<float4> _TagXZ;
Texture2D<float4> _TagXY;
Texture2D<float4> _TagYZ;
SamplerState sampler_point_clamp;
SamplerState sampler_linear_clamp;
float4 gsVoxCount;      // set globally by QuestRoomScan
float gsVoxSize;

// As ThermalKeyframes.Params.
struct Keyframe
{
    float4 row0, row1, row2;    // world -> RGB camera
    float4 position;            // RGB camera; w = tag
    float4 intrinsics;
    float4 crop;
    float4 mapA;
    float4 mapB;
    float4 info;                // x = time taken
};
#define KEYFRAMES 128
StructuredBuffer<Keyframe> _Keyframes;
Texture2DArray<float> _KeyframeTex;
float _Now;
float _PaintKeyframes;      // global (PerfMonitor): keyframes reprojected per pixel, newest first, 1..4
// Seconds over which an old keyframe's say halves against newer ones.
#define KEYFRAME_HALF_LIFE 60
#define SURFACE_MATCH 0.02      // same surface: position along the map's normal axis, volume units
#define NEARBY_TEXELS 4         // fallback search step, texels (~4 cm)

float2 MapUv(float2 baseUv, float normalSign)
{
    return float2(baseUv.x, baseUv.y * 0.5 + (normalSign > 0 ? 0.5 : 0));
}

// Adds one map's temperature, weighted, if it has data for this surface; the keyframes
// of the most weighted one are kept.
void Accumulate(Texture2D<float4> map, Texture2D<float4> tagMap, float2 baseUv, float normalSign,
                float along, float weight, inout float sum, inout float weights, inout float best,
                inout float4 tags)
{
    if (weight < 0.05) return;     // an axis the surface barely faces: skip its fetches
    float2 uv = MapUv(baseUv, normalSign);
    float4 t = map.SampleLevel(sampler_point_clamp, uv, 0);
    if (t.y > 0.01 && abs(t.z - along) < SURFACE_MATCH)
    {
        sum += t.x * weight;
        weights += weight;
        if (weight > best) { best = weight; tags = tagMap.SampleLevel(sampler_point_clamp, uv, 0); }
    }
}

// Texels around the point on one map, two rings of eight; the nearest ring with data wins.
void AccumulateNearby(Texture2D<float4> map, Texture2D<float4> tagMap, float2 baseUv, float normalSign,
                      float along, inout float sum, inout float weights, inout float4 tags)
{
    uint width, height;
    map.GetDimensions(width, height);
    float2 step = NEARBY_TEXELS / float2(width, height * 0.5);
    float best = 0;
    [loop] for (int ring = 1; ring <= 2 && weights < 1e-3; ring++)
    [loop] for (int k = 0; k < 8; k++)
    {
        float angle = k * (PI / 4);
        float2 offset = float2(cos(angle), sin(angle)) * step * ring;
        float4 t = map.SampleLevel(sampler_point_clamp, MapUv(baseUv + offset, normalSign), 0);
        if (t.y > 0.01 && abs(t.z - along) < SURFACE_MATCH)
        {
            sum += t.x;
            weights += 1;
            if (t.y > best) { best = t.y; tags = tagMap.SampleLevel(sampler_point_clamp, MapUv(baseUv + offset, normalSign), 0); }
        }
    }
}

// °C of a world point in a stored keyframe (bilinear, less the sensor offsets),
// and its weight: falls to 0 at the image edge, halves every KEYFRAME_HALF_LIFE.
// False when the keyframe was overwritten or the point falls outside its image.
// Same projection as ThermalLive.
bool KeyframeCelsius(float3 world, float tag, out float celsius, out float weight)
{
    celsius = 0;
    weight = 0;
    if (tag < 0.5) return false;
    uint slot = ((uint)round(tag) - 1) % KEYFRAMES;
    Keyframe k = _Keyframes[slot];
    if (abs(k.position.w - tag) > 0.25) return false;
    float4 p = float4(world, 1);
    float3 c = float3(dot(k.row0, p), dot(k.row1, p), dot(k.row2, p));
    float2 uv;
    if (!ThermalUvOf(c, k.intrinsics, k.crop, k.mapA, k.mapB, uv) || any(uv < 0) || any(uv > 1)) return false;

    // R16F filters in hardware: one fetch instead of four loads.
    celsius = _KeyframeTex.SampleLevel(sampler_linear_clamp, float3(uv, slot), 0) - FlatOffset(uv);
    float2 d = abs(uv - 0.5) * 2;
    float centrality = 1 - max(d.x, d.y);
    weight = centrality * centrality * exp2(-(_Now - k.info.x) / KEYFRAME_HALF_LIFE);
    return true;
}

// °C at a surface point with its normal; false where the room has no thermal data.
bool PaintCelsius(float3 world, float3 normal, out float celsius)
{
    celsius = 0;
    float3 uvw = world / (gsVoxCount.xyz * gsVoxSize) + 0.5;
    float3 n = normalize(normal);
    float3 w = abs(n) / (abs(n.x) + abs(n.y) + abs(n.z) + 1e-4);
    float sum = 0, weights = 0, best = 0;
    float4 tags = 0;
    Accumulate(_MapXZ, _TagXZ, uvw.xz, n.y, uvw.y, w.y, sum, weights, best, tags);
    Accumulate(_MapXY, _TagXY, uvw.xy, n.z, uvw.z, w.z, sum, weights, best, tags);
    Accumulate(_MapYZ, _TagYZ, uvw.yz, n.x, uvw.x, w.x, sum, weights, best, tags);
    if (weights < 1e-3)
    {
        // The axis the surface faces most.
        if (w.y >= w.x && w.y >= w.z) AccumulateNearby(_MapXZ, _TagXZ, uvw.xz, n.y, uvw.y, sum, weights, tags);
        else if (w.z >= w.x) AccumulateNearby(_MapXY, _TagXY, uvw.xy, n.z, uvw.z, sum, weights, tags);
        else AccumulateNearby(_MapYZ, _TagYZ, uvw.yz, n.x, uvw.x, sum, weights, tags);
    }
    if (weights < 1e-3) return false;
    celsius = sum / weights;

    // Newest first (tags grow with time; empty = 0 sorts last), so fewer keyframes keep the freshest.
    float2 p0 = float2(max(tags.x, tags.y), min(tags.x, tags.y));
    float2 p1 = float2(max(tags.z, tags.w), min(tags.z, tags.w));
    float4 newest = float4(max(p0.x, p1.x), 0, 0, min(p0.y, p1.y));
    float a = min(p0.x, p1.x), b = max(p0.y, p1.y);
    newest.y = max(a, b);
    newest.z = min(a, b);
    float frameSum = 0, frameWeights = 0;
    [unroll] for (int s = 0; s < 4; s++)
    {
        if (s >= _PaintKeyframes) break;
        float c, weight;
        if (KeyframeCelsius(world, newest[s], c, weight)) { frameSum += c * weight; frameWeights += weight; }
    }
    if (frameWeights > 1e-5) celsius = frameSum / frameWeights;
    return true;
}

#endif
