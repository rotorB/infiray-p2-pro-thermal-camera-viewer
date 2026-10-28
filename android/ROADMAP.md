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
     palettes, range modes; B = palette, right stick click = range, Y = recenter).
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

## Phase 2: calibration mode ("Train")

Goal: a mapping from a thermal pixel to an RGB pixel that holds at any distance, i.e. scale,
offset and the parallax caused by the two cameras sitting apart.

Flow:
1. User presses "Train" and looks at a rectangular target whose temperature differs clearly
   from the background (heated or cooled sheet, laptop lid, a box with a hand warmer behind
   it). On-screen hint explains this.
2. Each frame pair: find the target's 4 corners in both images.
   - Thermal: threshold against the background (Otsu on °C), largest contour, approximate to a
     quadrilateral, refine corners.
   - RGB: edges + quadrilateral search near the area predicted from the thermal corners (after
     the first good samples), refine corners.
   - Reject the pair if either quad is missing, too small, too skewed or blurred (motion).
3. Distance per sample: from the Depth API at the target, or from the target's apparent size
   once its physical size is known.
4. The user walks closer and further away; the app collects samples across distances and
   shows coverage (near / mid / far) and the current error.
5. Fit:
   - Model: thermal and RGB cameras as two pinhole cameras with fixed relative rotation R and
     translation t. For a point at depth d the thermal->RGB mapping is
     `H(d) = K_rgb (R + t nᵀ / d) K_th⁻¹`. Fit R, t (and refine the thermal intrinsics K_th,
     starting from the P2 Pro nominal FOV) over all corner correspondences.
   - Robust fit: RANSAC over samples, then least squares on the inliers; drop outlier frames.
   - Simple fallback if the full model is unstable: per-distance homographies, interpolated by
     depth.
6. Report reprojection error in RGB pixels; accept the calibration only below a threshold.
   Save it (with mount id, camera id and date) and load it on start.

Open: whether to also offer a checkerboard target (heated print) for a more precise
intrinsic calibration of the thermal camera.

## Phase 3: Supervision 1 (raster fusion)

Thermal image warped into the RGB camera's view with the Phase 2 mapping, then blended onto the
RGB image.

- [ ] Warp: per frame, using the depth at the scene (single dominant depth to start, per-pixel
  depth from the Depth API later).
- [ ] Thermal source as palette colour or as greyscale (hotter = whiter).
- [ ] Blend modes, switchable live:
  - opacity (alpha slider): base mode;
  - add, multiply, screen, subtract, difference, overlay;
  - "hot only": thermal shown only above a temperature threshold.
- [ ] Controls: mode, opacity, palette, temperature range, threshold.
- [ ] Rendering: start in the 2D panel (RGB frame + overlay, accepts some latency); move to an
  MR layer over passthrough later if latency matters.

## Phase 4: Supervision 2 (thermal room mesh)

The headset builds a mesh of the room while the user walks; each mesh element gets the
temperature measured when it was seen.

- [ ] Live geometry: fuse Depth API (environment depth) frames with the head pose into a TSDF
  voxel grid, extract the mesh with marching cubes. Horizon OS's Scene mesh is static (from
  Space Setup), so live scanning has to be built on the Depth API.
- [ ] Thermal projection: for every new frame, project mesh vertices into the thermal camera
  (head pose + Phase 2 extrinsics); write the temperature to vertices that are visible
  (depth test against the current depth map) and well inside the thermal FOV.
- [ ] Store temperature, not colour: vertex attributes `temperature`, `last_seen`,
  `sample_count`; average or keep the latest per vertex (decide which).
- [ ] Colour mesh edges/faces from temperature with the selected palette; palette and range
  change live without rescanning.
- [ ] Start/stop scan, clear, export (PLY with a temperature attribute).

## Open decisions

- Which RGB camera (left/right) and resolution to standardise on.
- The thermal camera mount design and how to detect that it moved (re-check calibration error
  on start).
