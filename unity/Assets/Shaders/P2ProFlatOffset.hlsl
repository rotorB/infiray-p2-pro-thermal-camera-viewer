// Learnt sensor offsets of the P2 Pro (ThermalFlatField, P2ProFlatField.compute), subtracted
// where temperatures land on the room: when P2ProThermalBake.compute writes the map and when
// P2Pro/ThermalMesh reads a stored keyframe. The live image is never corrected.
#ifndef P2PRO_FLAT_OFFSET_INCLUDED
#define P2PRO_FLAT_OFFSET_INCLUDED

#define FLAT_GRID_X 32
#define FLAT_GRID_Y 24
StructuredBuffer<float> _FlatOffsets;   // °C per cell, row 0 at the bottom of the image
float _FlatOn;

float FlatCell(int2 c) { return _FlatOffsets[clamp(c.y, 0, FLAT_GRID_Y - 1) * FLAT_GRID_X + clamp(c.x, 0, FLAT_GRID_X - 1)]; }

// Offset (°C) at a thermal texture coordinate, bilinear between cell centres; 0 while off.
float FlatOffset(float2 uv)
{
    if (_FlatOn < 0.5) return 0;
    float2 p = uv * float2(FLAT_GRID_X, FLAT_GRID_Y) - 0.5;
    int2 i = (int2)floor(p);
    float2 f = p - i;
    float bottom = lerp(FlatCell(i), FlatCell(i + int2(1, 0)), f.x);
    float top = lerp(FlatCell(i + int2(0, 1)), FlatCell(i + int2(1, 1)), f.x);
    return lerp(bottom, top, f.y);
}

#endif
