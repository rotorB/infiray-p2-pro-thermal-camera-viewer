# P2 Pro Thermal for Quest 3: roadmap

Target product: a Quest 3 tool that shows the Infiray P2 Pro thermal image, registers it to the
headset's own RGB cameras, and fuses the two, first as a 2D raster overlay (Supervision 1), then
as a temperature-coloured 3D mesh of the room (Supervision 2).

Status legend: [x] done, [~] in progress, [ ] not started.

## Rule: temperatures travel, colours are applied at the end

Thermal data leaves the camera and moves through every stage (calibration, raster fusion, mesh
vertices, textures, export) as temperature in °C, never as palette colours. A palette is applied
only in the final shader that draws it, so palette and range can change live and the measured
values are never lost. Any buffer, texture or vertex attribute that carries thermal data holds
°C (float/half), not RGB.

## Decision: Unity for everything after Phase 0 (2026-10-28)

Engine: Unity 6.2+ (URP) with Meta XR SDK, OpenXR + Meta OpenXR, AR Foundation and MR Utility
Kit. Reasons:

- Depth API (needed for live meshing and per-pixel depth in the overlay) is supported in Unity;
  no Spatial SDK equivalent was found.
- [QuestRoomScan](https://github.com/genesisinteractive/QuestRoomScan) (MIT, Unity 6000.2,
  `com.genesis.roomscan` 1.3.2) already does live room reconstruction on Quest 3: GPU TSDF
  integration, GPU Surface Nets meshing, per-vertex colour and a triplanar colour cache fed from
  the passthrough RGB camera (`PassthroughCameraProvider`). Supervision 2 = this pipeline with a
  thermal camera provider in place of the RGB one.
- Accuracy does not depend on the engine: both use the same OS APIs. It depends on depth map
  quality, the Phase 2 calibration and frame time sync.

The Kotlin app in `android/` stays as the reference thermal viewer and as the source of the
native capture code.

## Unity: first steps

1. Install Unity Hub and Unity 6.2 (6000.2.x) with Android Build Support (SDK, NDK, OpenJDK).
2. New URP project in `unity/` in this repo. Add Meta XR SDK core, MR Utility Kit 205+,
   OpenXR, Meta OpenXR, AR Foundation 6.1+, and QuestRoomScan pinned to `#v1.3.2`.
   Done on Unity 6000.3.27f1 LTS. QuestRoomScan 1.3.2 is embedded in
   `unity/Packages/com.genesis.roomscan` (copied from tag `v1.3.2`, `docs/` left out): its setup
   wizard calls Unity 6000.4+ APIs (`FindObjectsByType` without a sort mode, top-level
   `WorldSpaceSizeMode`), patched in `Editor/RoomScanSetupWizard*.cs` to build on 6000.3. The
   embedded copy is also the base of the thermal fork.
3. Baseline: build QuestRoomScan's game-ready setup to the Quest and confirm scanning works with
   no thermal code. This proves the toolchain, permissions and depth before anything is ours.
   Done 2026-10-28: scene `Assets/Supervision.unity`, live mesh in the headset at ~70 depth
   frames/s, wireframe by default, X cycles render modes. Notes:
   - the game-ready preset does not start a scan: `Assets/Scripts/AutoStartScan.cs` calls
     `RoomScanSession.StartScanAsync()` (which also requests the permissions);
   - the OpenXR features `Meta Quest: Session`, `Occlusion` and `Camera` must be on for Android,
     otherwise there are no depth frames; `Assets/Editor/QuestBuild.cs` enables them;
   - build, install and start from the command line with `unity/build.sh <adb-serial>` (editor
     closed). It runs Unity with a minimal PATH because a broken `/usr/local/bin/cmake` on this
     Mac breaks Gradle's CMake step.
4. Thermal Android plugin:
   - reuse `raw_uvc.c` and the libuvc it loads (UVCAndroid AAR in `Plugins/Android`, or libusb +
     libuvc built from source);
   - request USB permission with `UsbManager` directly (Unity targets SDK 33+, where
     UVCAndroid's `USBMonitor` broadcast breaks: needs an explicit, mutable `PendingIntent`);
   - keep the kernel-driver detach and `horizonos.permission.USB_CAMERA`.
5. `ThermalCameraProvider` (C#): frames from the plugin -> temperature texture (`RFloat`,
   256x192, °C) + frame timestamp. A world-space panel with palette/range shows it: parity with
   the Kotlin viewer is the first Unity milestone.
   Steps 4-5 done 2026-10-28 (25 fps in the headset, room 19.5-26 °C):
   - `android/native/raw_uvc.c` is shared by the Kotlin app and the Unity plugin; its plain C
     API (`p2pro_open`, `p2pro_take_celsius`, `p2pro_close`) is called from C# by P/Invoke;
   - `android/unityplugin` (AAR, Java only): `UsbThermal` finds the camera, requests USB access
     with an explicit mutable `PendingIntent`, detaches the kernel driver and hands the fd to
     Unity; the AAR also carries UVCAndroid's `libuvc`, `libusb1.0`, `libjpeg-turbo212`;
   - Unity: `ThermalCamera` (permissions, stream, RFloat texture), `ThermalPanel` (panel,
     palettes, range modes; B = palette, right stick click = range; Y now cycles the room mode).
   Still missing for parity: on-panel min/max/center readout and markers, rotate/mirror,
   frame timestamps for pairing with RGB and depth.
6. Then Phase 1-4 below, implemented in Unity. For Phase 4, store temperature (not colour) in
   QuestRoomScan's vertex/triplanar layers and map it through the palette in the shader.

## Phase 0: thermal viewer (current)

- [x] UVC access on Horizon OS: `horizonos.permission.USB_CAMERA`, kernel `uvcvideo` detach.
- [x] Decode YUYV 256x384, bottom half = uint16 LE, `°C = raw / 64 - 273.15`.
- [x] Palettes, range modes, min/max/center markers, pointer temperature, rotate/mirror.
- [x] Byte-exact capture through libuvc (`RawUvc`, `raw_uvc.c`). UVCAndroid's frame callback
  round-trips YUYV -> RGBX -> YUYV and destroyed the temperature half.
- [x] Verified on Quest (2026-10-28): 25 fps, room 21.6-23.5 °C, stable over 500 frames.
- [ ] Visual check in the headset: palm ~33-35 °C, no column stripes.
- [ ] Update README (capture path), commit.

Note: temperature checks on the Mac must run from Terminal. Capture started from inside
Claude.app returns altered frames (per-app camera processing in macOS), the same command
from Terminal returns correct data.

## Phase 1: second source, the headset RGB camera

Done in Unity 2026-10-28 (`Assets/Scripts/Thermal`), verified in the headset:

- [x] RGB frames from MRUK's `PassthroughCameraAccess` (the one QuestRoomScan runs): left
  camera, 1280x960, ~27 fps (`HeadsetRgbCamera`).
- [x] Lens intrinsics (focal length, principal point, sensor resolution) and the lens pose
  relative to the head (`Intrinsics.LensOffset`); camera pose at exposure time.
- [x] Both streams at once. Common time base: thermal frames carry CLOCK_MONOTONIC arrival
  time from `raw_uvc.c`; RGB frames carry PCA's monotonic exposure time. Measured: XrTime -
  CLOCK_MONOTONIC = 0.0 ms, so both are XrTime. Head pose at any frame time from
  `OVRPlugin.GetNodePoseStateAtTime`.
- [x] Pairing by nearest time (`FramePairing`, 16-frame thermal ring buffer, °C only): no
  misses, |offset| mean ~10 ms, max ~20 ms (half a thermal frame, as expected at 25 fps).
- [x] Debug view: thermal (left) and RGB (right) side by side.

Notes for Phase 2:
- Thermal time is USB arrival, not exposure: `FramePairing.ThermalLatencyMs` (now 0) should be
  estimated in calibration (fit the time offset that minimises reprojection error while moving).
- OVRPlugin head pose + `LensOffset` differs from PCA's own `GetCameraPose()` by ~2 mm, 0°.
  Use one pose source for both cameras (OVRPlugin head pose at each frame's time) so the
  thermal extrinsics are fitted against the same reference.

Precondition for everything below: the P2 Pro is mounted rigidly on the headset. Any shift of
the mount invalidates the calibration.

## Phase 2: alignment ("Align" mode, replaces "Train")

Done in Unity 2026-10-28 (`Assets/Scripts/Alignment`), verified in the headset: one point was
enough for a good overlay.

The full two-camera model (R, t, K_th) was replaced by a simpler 2D similarity per distance:
thermal UV -> RGB viewport as offset (U, V), scale and roll, each fitted as `A + B / distance`
(parallax). `ThermalAlignment` stores points in `persistentDataPath/thermal_alignment.json`.

- [x] Align mode on the panel (virtual buttons, pointer rays from both controllers):
  Calibrate -> Capture freezes both images; the stick moves, scales and rolls the thermal
  overlay; Save point stores it with the distance from a depth ray march (`DepthProbe`).
- [x] Automatic start: `TargetFinder` finds a warm/cold rectangle in both images (Otsu) and
  `FitCorners` fits the similarity; the user corrects it by hand before saving.
- [x] Undo last point, Reset all points (two-click confirm), plausibility checks on points.
- [x] Thermal camera roll (mount rotation) is part of the fit.
- [x] Latency: `LatencyEstimator` compares the measured image shift between thermal frames
  (Lucas-Kanade) with the shift predicted from head poses at 0-200 ms; auto or manual
  (+-10 ms buttons).
- [ ] More points at near / mid / far distances to check the 1/d model.

## Phase 3: Supervision 1 (live thermal on the room)

The latest thermal frame is projected per fragment onto the room mesh (`ThermalLive`,
P2Pro/ThermalLive) through the RGB camera pose at exposure time and the alignment, instead of
warping a 2D image.

- [x] Room modes: Live (latest frame only), Paint (Phase 4), Off; Menu hides/shows the panel.
- [x] Palette and range modes: Frame, Room, Wide, Scene (2-98 % of all stored °C).
  Ten palettes (Inferno, Turbo, Ironbow, White / Black hot, Rainbow, Arctic, Lava, Viridis,
  Hot spots).
- [x] Controller shortcuts: Y = room mode (live, paint, flat), X = thermal surfaces / thermal
  wireframe (hair-thin, antialiased, coloured by °C; starts on surfaces), right stick click =
  palette, left stick click = range (Scene, Frame, Wide), B = HUD (modes, readings, palette
  bar; follows the head, remembered).
- [x] QuestRoomScan's own mesh (RGB colours, white grid) is hidden; a depth-only pass of the
  same mesh keeps the thermal overlays on the nearest surface.
- [x] Overlay style (`OverlayStyle`, `P2ProOverlay.hlsl`), saved between launches:
  - blend: alpha, add, screen, multiply;
  - opacity slider (10-100 %);
  - show all, only above or only below a threshold (slider over the palette range; dragging
    it with the filter off turns on hot-only);
  - greyscale through the White hot / Black hot palettes.
- [x] Pointer sliders in `ButtonBar` (point, hold the trigger, drag): threshold, opacity,
  palette min / max (switch the range to Manual, remembered; replaces the fixed 15-35 range).
- [x] Panel tabs (remembered): View (room mode, clear map, palette, range, palette min / max),
  Style (blend, show filter, wire, threshold, opacity, sharpness 0-10 with 0 = off),
  Setup (calibrate, recenter, even, latency). At most three rows.
- [x] Thermal flashlight (`ThermalFlashlight`): A toggles it; while on, Live, Paint and Flat
  show thermal only inside a soft-edged cone from the right controller (half-angle 3-45 deg,
  slider in the View tab). Starts off every launch.
- [x] Edges look (X cycles surface -> wire -> edges; `P2ProThermalEdges.hlsl`): Live and Flat
  draw only the thermal image's edges, 1 camera pixel wide (gradient ridge, non-maximum
  suppression), steeper than a threshold (slider, °C per pixel). Lines on real outlines =
  good alignment; step 1 of the edge-based auto alignment.
- [x] Edge align, experimental (`EdgeAlign`, P2ProEdgeAlign.compute; Setup -> Edge align):
  the Edges look also draws the RGB camera's edges in cyan (white where both meet). While
  the head is still (0.3 deg / 5 mm over 0.3 s), one frame a second is collected: RGB
  luminance edges sampled 4 x 4 per environment depth pixel (away from depth jumps) as 3D
  points in the RGB camera frame at the thermal frame's time, and the thermal frame's edge
  ridges (sub-pixel, nearest-edge map). Solve (background): ICP point-to-line with Huber
  weights, gates 8 -> 1.5 px, fitting rotation, fx, fy, cx, cy, k1, k2 (position kept);
  then a 16 x 12 grid of offsets (least squares across edge normals, smoothed) applied after
  the lens in P2ProThermalLens.hlsl and `ThermalLens`. Scored per zone (centre / middle /
  rim: median px across the edge, share within 1 px) against the current alignment; Apply
  only when better, and it clears the map.
- [x] Edge align, round 2:
  - RGB edges at the thermal camera's pixel size (`RgbEdgeMap`, P2ProRgbEdgeMap.compute):
    the image is averaged down by the focal length ratio before the gradient, so text and
    pictures on a screen fade and object outlines stay; threshold slider. Depth jumps in the
    collector use the nearest depth, so outlines in front of a far wall count.
  - Calibrations kept side by side: mapping, lens (points), edge, edge + grid; "Using:" picks
    the one that projects (Setup and Edge align), so each can be judged alone. The panel
    overlay follows the one in use (`ThermalAlignment.TryOverlay`).
  - Mount roll (0 / 90 / 180 / 270 deg), the root of every projection: each calibration keeps
    the mount it was made with and is turned by the difference (`ThermalLens.Remounted`, the
    mapping's roll), so changing it turns the thermal image at once. With no calibration, or
    "Using: Nominal", the nominal lens turned by the mount projects, so Edge align can start
    from nothing ("Reset calibrations" deletes mapping, lens and edge; the mount stays).
  - Buttons that cannot act now are greyed and say why when pressed; every press flashes and
    is echoed in the status and the HUD.
  - Telemetry: still / moving, frames, last frame's points, the live fit of the calibration
    in use per zone, result before -> after. Panel images captioned (which camera, live or
    frozen), with thermal edges magenta and RGB edges cyan during Edge align.
- [x] Edge align is the calibration: it beat the mapping and the points lens everywhere,
  corners included, from one look at a TV's outline. Setup offers only Calibrate (edge),
  Mount and Reset; the mapping and the wizard stay in code and file without buttons. An edge
  choice falls back to the nominal lens, never to the old calibrations.
- [x] Controller guide (`ControllerGuide`): all controller buttons in one table. A press
  shows a toast in the middle of the view (button, what it switches, every choice with the
  new one in brackets, or why not now); looking at a controller shows its buttons with their
  settings above it. The panel's status starts with where it is (menu tab or calibration).
- [x] Performance (`PerfMonitor`, Setup tab; GPU time from VrApi "App=" in logcat). Measured
  GPU-bound, CPU ~4 ms. Fixes: URP intermediate texture Auto (was Always: a full-res copy per
  eye, no foveation); Meta foveation via OVRManager, default high (Paint 18 -> 13 ms); room
  scan 3 integrations / 1 mesh rebuild per second by default (was 30 / 8: Flat 72 -> 86 fps,
  60-90 ms hitches gone); Paint reprojects the 2 newest of 4 keyframes per pixel by default
  (4: 18.6 ms GPU, 2: 11.7, 1: 10.3), each with one hardware-filtered R16F fetch, and skips
  triplanar axes the surface barely faces (1 keyframe: 10.3 -> 5.9 ms). At 90 Hz: Live 87,
  Paint 86, Flat 88 fps; the rest are the scan's 3 + 1 GPU bursts a second.
  Edge scoring only while calibrating. Switches for foveation, resolution, MSAA, display rate,
  scan rate, paint keyframes.
- [x] Shutter (flat-field calibration): while the camera's shutter is closed (11-12 frames,
  ~0.5 s, every ~15 s while warming up, 60 s later) it repeats its last frame with fresh arrival
  times, so the old picture was projected with newer head poses and slid across the room.
  Dropping the repeats and holding frames after them did not stop the slide, so the automatic
  shutter can be switched off instead: `set_prop_auto_shutter_params` (0xc214, PROP_SWITCH = 0,
  read back with 0x8214; from stanelie/p2-viewer, extracted from the official app's
  libircmd.so) through the vendor control channel in `raw_uvc.c`. Written 12 s after every
  stream start, after the camera's two startup calibrations (it forgets the switch when it loses
  power; nothing is written to flash). Setup tab switches it back on. Nothing ever triggers a
  calibration. Tried off: no more clicks, but temperatures drifted as the sensor warmed, so
  auto stays the default and off is a Setup choice.
- [x] Paint holes and lumpy walls. Paint vanished where the scan mesh was rebuilt: each depth
  point was baked into one map only (its dominant normal axis), and a triangle that came back
  moved or turned found no matching texel and was discarded. Now tilted points (> ~24 deg) go
  into the second map too, and a pixel with no texel under it takes the texels ~4-8 cm around
  (P2ProThermalPaint.hlsl). Tried and removed: walls, floors and ceilings as Space Setup
  (MRUK) planes over the scan mesh. Every mesh fragment tested itself against all planes, and
  the planes overdrew the view: far too slow on the headset.
- [x] Scan tuning (`ScanTuning`, Setup tab): voxel size slider 2.5-8 cm (RoomScan patch
  `SetVoxelSize`: same 384 x 128 x 384 grid, so finer voxels cover a smaller volume; the scan
  and the painted room restart). The world is now shifted so the floor sits 1.3 m below the
  volume's centre, which keeps a 2.9 m ceiling inside at 2.5 cm. Freeze seen surfaces: once a
  second, voxels in a 35 deg cone up to 4 m that reached 90 % of the full weight are frozen
  (RoomScan patch: weighted `FreezeInView`, `UnfreezeAll`), so converged geometry stops being
  rebuilt.
- [x] Temperature band and UI pass: the colour filter is a band (`OverlayStyle.BandLow` ..
  `BandHigh`) coloured alone (Inside) or left out (Outside). The HUD is a rounded card
  (`UiRect`, P2Pro/UIRect: SDF rounded rectangles) with rich-text status, the palette bar over
  the colour range with two draggable band handles, and an All / Inside / Outside switch. The
  panel shrank (images 0.25 m, buttons 3 cm on a dark card) so it fits the view at arm's length.
- [ ] Subtract, difference, overlay blend modes, if needed.

## Phase 4: Supervision 2 (thermal room mesh)

Geometry comes from QuestRoomScan (GPU TSDF + surface nets, `GPUMeshRenderer` draws overlay
materials over the same buffers).

- [x] Persistent thermal texture (`ThermalMap`, P2ProThermalBake.compute): each thermal frame
  is splatted from environment depth into three world-space triplanar maps (2048², RGBA16F,
  texel = °C, confidence, axis position). Surfaces in view keep updating, the rest keep their
  last value; blend weight depends on viewing angle and distance.
- [x] Full thermal resolution on the painted room (`ThermalKeyframes`): the map's texels
  (9 x 19 mm over a 19.2 m volume) are too coarse, so each texel also names the keyframe that
  last saw it, and the mesh shader projects that whole stored frame like the live view.
  128 immutable keyframes (R16F): a new one after 3 cm / 2 deg of head motion or 1 s still,
  at most 5 a second; the slot reused is the one fewest texels still name (counted with the
  Scene histogram). Texels whose keyframe was replaced fall back to the map's °C.
- [x] Even painting across the sensor: a texel keeps up to four keyframes (a second set of
  triplanar tag maps, +96 MB); a new keyframe replaces the weakest (score = centrality in that
  image less 0.01 per second of age). The mesh shader averages the four stored frames per
  fragment, weighted by centrality² (0 at the image edge, so views fade into each other) and a
  60 s half-life. The map's running °C weights the image rim down to 5 %. Only what lands on
  the room is averaged: the live thermal image is never filtered.
- [x] Flat-field correction (`ThermalFlatField`, P2ProFlatField.compute): a 32 x 24 grid of
  offsets learnt from the room (frame minus the map's running average where that is reliable
  and the image is locally flat), 2 % per frame, image centre anchored at 0; subtracted only
  where temperatures land on the room (map writes in the bake, keyframe reads in the mesh
  shader, P2ProFlatOffset.hlsl). Panel, Live and Flat show the camera's frame unchanged. Off by default: without a restoring
  force the offsets drifted to a 7.5 C spread and brightened the painted image edges; now they
  decay 0.2 % a frame and are capped at 1 C. "Even" button = on/off (off -> on relearns); the log prints the left / centre / right offsets every 10 s.
- [x] The map keeps a running average per texel (first view as is, then weighted, capped at
  confidence 10) instead of following each frame.
- [x] Pointer label drawn over everything (it was hidden by the overlay's sort order).
- [x] Thermal lens model (`ThermalLens`, P2ProThermalLens.hlsl): the thermal camera as a real
  camera in the RGB camera's frame (rotation, position, fx, fy, cx, cy, k1, k2), fitted by
  Levenberg-Marquardt to point pairs from the Points wizard (`PointsWizard`, Calibrate ->
  Points wizard): mark a feature on the thermal image, then on the RGB image; the RGB point's
  3D position comes from the room depth. Rotation from four starting rolls, then intrinsics,
  then position (pairs spanning distances) and distortion; outliers trimmed once. Once solved
  it replaces the A + B / distance mapping in the bake, keyframes, Live, Flat (a 16 x 12
  grid) and the latency estimate; stored with the alignment. In the wizard the thermal image
  is shown turned like the RGB one (mapping roll), plus "Turn thermal" quarter turns
  (remembered); display only.
- [x] USB device filter (VID 0x0BDA / PID 0x5830) on the activity, so the system can remember
  "always open" for the P2 Pro instead of asking for USB access every time.
- [x] Store temperature, not colour; palette and range change live without rescanning.
- [x] Clear map button; Scene range from a GPU histogram of the map twice a second.
- [ ] Known limit: two parallel surfaces facing the same way share triplanar texels.
- [ ] Save/load the scan, export (PLY with a temperature attribute).

## Open decisions

- Which RGB camera (left/right) and resolution to standardise on.
- The thermal camera mount design and how to detect that it moved (re-check calibration error
  on start).
