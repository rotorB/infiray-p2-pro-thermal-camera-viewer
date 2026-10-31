# Infiray P2 Pro thermal camera: Quest 3 and macOS

Two apps for the **Infiray P2 Pro** USB thermal camera (256×192, 25 fps):

- **[Quest 3 mixed-reality app](#quest-3-thermal-room-app)**: the camera plugs into a Meta Quest 3 and its temperatures are projected onto a live 3D scan of the room. Heat stays on walls, pipes and windows as you walk around.
- **[macOS desktop viewer](#macos-desktop-viewer)**: a full thermal viewer with palettes, measurements, super-resolution and a virtual webcam.

## Where things are

| Path | What it is |
|------|------------|
| [`unity/`](unity) | Quest 3 app (Unity 6, URP, Meta XR SDK). Build with `unity/build.sh`. |
| [`unity/Assets/Scripts/`](unity/Assets/Scripts) | App code: `Thermal/` (camera, panel, UI), `Alignment/` (calibration, projection, paint, HUD). |
| [`unity/Packages/com.genesis.roomscan/`](unity/Packages/com.genesis.roomscan) | [QuestRoomScan](https://github.com/genesisinteractive/QuestRoomScan) 1.3.2, embedded and patched (room mesh from the Quest depth sensor). |
| [`android/native/raw_uvc.c`](android/native/raw_uvc.c) | Native USB capture of the P2 Pro (libuvc), shared by the Quest apps. |
| [`android/unityplugin/`](android/unityplugin) | Android plugin (AAR): USB permission and device hand-over to Unity. |
| [`android/app/`](android/app) | Earlier standalone Kotlin viewer for Quest: a flat thermal panel, no 3D ([README](android/README.md)). |
| [`android/ROADMAP.md`](android/ROADMAP.md) | Design notes and history of the Quest app: what was tried, measured and why. |
| [`app.py`](app.py) and other `*.py` | macOS desktop viewer. |

---

## Quest 3 thermal room app

### What it does

- **Live:** the current thermal frame drawn on the surfaces you look at.
- **Paint:** walk around and cover the room in temperatures; they stay on the 3D surfaces after you look away, at the camera's full resolution.
- **Flat:** a plain thermal image over passthrough.
- Palettes and colour ranges switchable live; a temperature band filter (show only a range, or everything outside it); a thermal flashlight that shows heat only where the controller points.
- Automatic calibration against the Quest's own RGB camera, and automatic latency estimation.
- Runs on the headset alone, ~86-88 fps at 90 Hz.

How it works, in short: a native plugin reads raw sensor data over USB; the Quest depth sensor builds a live room mesh (QuestRoomScan); every pixel of the mesh is projected into the thermal image using the headset pose at the moment the frame was taken. Calibration matches edges in the thermal image with edges in the Quest RGB camera. Details in [`android/ROADMAP.md`](android/ROADMAP.md).

### What you need

- Meta Quest 3 in developer mode, and `adb`.
- Infiray P2 Pro, fixed rigidly to the headset (double-sided tape works) and plugged into the headset's USB-C port.
- A Mac with **Unity 6000.3.27f1** (Unity Hub, with Android Build Support), **JDK 17** and the **Android SDK** (for the plugin build).

### Build and install

1. The camera takes the headset's only USB-C port, so connect ADB over Wi-Fi: with the cable in, run `adb tcpip 5555`, then `adb connect <headset-ip>:5555`.
2. Point the Android build at your SDK: `echo "sdk.dir=<path-to-android-sdk>" > android/local.properties`.
3. If your Unity is not at `/Applications/Unity/Hub/Editor/6000.3.27f1`, edit `UNITY=` in `unity/build.sh`.
4. Close the Unity editor, then build, install and start:

   ```bash
   cd unity && ./build.sh <headset-ip>:5555
   ```

   The script builds the plugin AAR, builds the APK in batch mode, installs it and starts it.

### First run

1. Plug the P2 Pro in and allow camera, USB and spatial data access when asked.
2. Open the panel (**Menu** button), **Setup** tab: set **Mount** until the thermal image stands upright.
3. **Calibrate**: look at something with clear outlines and a temperature contrast (a TV that is on works well) and hold your head still for a moment.
4. Shake your head left and right for a few seconds so the latency estimate settles.

### Controls

| Button | Action |
|--------|--------|
| Y | Room mode: Live / Paint / Flat |
| X | Look: surfaces / wireframe / edges |
| Left stick click | Colour range |
| Right stick click | Palette |
| A | Thermal flashlight |
| B | HUD (status, band slider) |
| Menu | Panel with tabs (View, Style, Setup) |
| Trigger | Point to read a temperature; press panel and HUD controls |

Look at a controller to see what its buttons do right now.

---

## macOS desktop viewer

![Demo](demo_400.gif)

Real-time thermal imaging desktop app. Captures the raw YUYV stream (256×384), extracts the 256×192 thermal layer, converts to temperature, and displays with colormaps, measurement tools, and optional analysis overlays. Optimized for **macOS** (AVFoundation + ffmpeg); includes system tray and optional PyInstaller build.

### Requirements

- **Hardware:** Infiray P2 Pro (or compatible USB thermal camera: PureThermal, "USB Camera" with same protocol).
- **System:** macOS (AVFoundation). The standalone app includes **ffmpeg**. For running from source or building, install it first (e.g. `brew install ffmpeg`).
- **Python:** 3.9+ recommended (3.10 in project venv). Dependencies: `opencv-python`, `numpy`, `pystray`, `Pillow`, `pyvirtualcam>=0.14` (for virtual webcam; 0.14+ needs Python 3.9+).

---

### Quick Start

```bash
pip install -r requirements.txt
python app.py
```

For capture/temperature troubleshooting, run `venv/bin/python app.py --baseline --diagnostics`.
This starts in Celsius with the ordinary Turbo palette, with SR, DYN, DDE,
denoising and offset corrections disabled. It does not overwrite saved app
settings. Diagnostics report the selected script/ffmpeg and sensor vs corrected
temperature ranges plus the cursor temperature; a plausible numeric range alone
does not establish measurement accuracy.

Connect the P2 Pro; the app auto-detects it and opens the viewer. Use **I** in-app for the full control list.

If the camera is plugged in but the viewer stays on **Reconnecting...**, macOS has not published it as a camera. Chrome/Brave WebUSB often holds the USB device so UVC never binds — close those browsers, or unplug the P2 Pro for ~5 s and plug it back in. The overlay names the process that has the device open.

---

### Functionality Overview

#### Core imaging

- **Live thermal stream** — 256×192 thermal data, 3× upscale to 768×576, 25 FPS.
- **Temperature** — Raw 16-bit → Celsius (formula: `/64 - 273.15`), clip -20…150°C for display.
- **Range presets (0):** Auto (frame min–max), Room (15–35°C), Wide (-20–150°C).
- **Palettes:** **P** — cycle through all; **1–9** — jump to palettes 1–9; **! @ # $ % ^ &** (Shift+1…7) — jump to palettes 10–16.

#### Image quality

- **Color + details (DYN / `{`):** enabled by default with Turbo on first use. Preserves the palette's temperature-coded hue while lifting dark colors and reserving highlight headroom. Local float-temperature detail modulates perceptual lightness at two spatial scales, with edge-aware filtering, a soft noise floor and bounded gain. Chroma is reduced where needed to fit the display gamut. **C− / C+** or **[ / ]** adjust gain 0–4 (default 2); **( / )** always adjust gain even with RGB enabled, where [ / ] remain zoom. Works with SR, other palettes, inversion, Delta-T, snapshots and frozen-frame comparison. Bypasses legacy DDE contrast/sharpening while enabled. Auto range endpoints are gently smoothed; image content is never temporally averaged by this mode. Temperature readings stay native; enhanced lightness is not an absolute temperature scale. DYN at gain 0 keeps palette illumination; turn DYN off for the ordinary palette/legacy DDE. `--dynamic-contrast` starts with DYN and Turbo explicitly.
- **Float pipeline:** reconstruction uses original float °C samples before display denoising/clipping. Cursor, ROI, min/max and profiles use native temperature measurements; reconstructed pixels are display estimates. 8-bit mapping is only for display.
- **DDE – High Quality (H):** contrast (gamma / percentile / CLAHE) + mild unsharp, applied *after* upsample. **[ / ]** = contrast, **-** **=** = sharpness.
- **IBP upscale (Y):** Iterative Back-Projection on the float field; **J** / **K** = strength 1–10.
- **Adaptive SR 2x (E / SR2x):** enabled by default; jointly reconstructs a 512×384 field from 8–16 original frames in one background worker, displayed at the usual 768×576 size. Only extra detail in stable, confidently aligned regions is added to the newest frame. Moving objects, thermal changes and their boundaries immediately use the current single frame. Camera jumps, capture gaps and calibration changes reset history; cancelled results cannot return. **E** compares on/off and is saved between launches. G/Y select the separate legacy denoise/IBP paths and turn adaptive SR off; E turns those paths off. Extra DDE sharpening and RGB edge guidance are bypassed while SR is active.
- **SR status:** `SR collecting` / `SR working` while preparing; `SR need microshift` when the frames lack independent 2D subpixel positions; `SR 2x N%` reports the fraction of the displayed area accepting reconstructed detail. Gently move the camera in two directions while the subject stays still. A perfectly stationary camera only repeats the same sampling grid. `2x` describes the reconstruction grid, not a measured doubling of optical resolution.
- **HQ snapshot (S):** saves the current view and a separate `_hq.png` if a validated reconstruction is ready. It never starts a blocking solver. Freeze with **Space**, wait briefly for an in-flight job if necessary, then **S**. Unsupported or moving regions retain the latest native-frame appearance.
- **RGB-guided edges (O):** when the RGB camera is aligned, a guided filter snaps thermal edges to RGB edges without copying RGB texture (a poster on a wall does not become fake heat). Blend (full / hot-mask) is unchanged.
- **FPN correction:** **C** = scene-based NUC. Learns column/row stripes and per-pixel offsets only while the camera pans (NCC-gated LMS; a static scene or a uniform wall is never burned in). **X** = flat-field: aim at a uniform surface, hold still ~2 s; a median of frames minus a fitted plane becomes the offset map (saved to disk, restored on launch). Moving during **X** is skipped (progress stalls). The target is also checked for uniformity: calibrating on a textured scene is refused (`FFC failed - aim at a uniform surface`) instead of baking a ghost of that scene into the map. A stored map that fails the same check is not loaded. **X** during calibration cancels, **X** while on = off.
- **De-noise (G):** motion-compensated Kalman (static pixels average down NETD, moving pixels keep the new sample) + Edge-Gated Local Mean. **,** **.** = strength.

#### Orientation (rotate & flip)

- **Left / Right arrows** — Rotate image by 90° (0°, 90°, 180°, 270°).
- **Up arrow** — Flip vertically (mirror top–bottom).
- **Down arrow** — Flip horizontally (mirror left–right).

Useful when the camera is mounted upside-down or sideways; all overlays and measurements follow the transformed view.

#### Measurements

- **Cursor** — Green crosshair + temperature at mouse position.
- **Min/Max markers (M)** — Blue (min) and red (max) triangles with values.
- **ROI (R)** — Draw rectangle; shows min / avg / max for the region.
- **Line profile (L)** — Draw a line on the image; panel shows temperature graph along the line (draggable/resizable panel).
- **Delta-T (V)** — Capture reference frame; display shows temperature difference (diverging colormap). Cursor shows absolute + delta. **V** again = reset.

#### Analysis overlays

- **Isotherms (T)** — Contour lines at 6 temperature levels with labels.
- **Histogram (B)** — Temperature distribution in a draggable/resizable panel.
- **Trend (W)** — Cursor temperature over time (last 300 samples) in a draggable/resizable panel.
- **Anomaly detection (N)** — Z-score (mean + 2σ); highlights hot regions and shows count + threshold.

#### Panels (Trend, Histogram, Line profile)

- **Drag** by the title bar; **resize** by the bottom-right grip.
- Layout is saved to `~/Library/Application Support/Infiray P2 Pro/layout.json` and restored on next run.
- White border indicates drag state.

#### Other

- **Rotate & flip** — **←/→** rotate 90°; **↑** flip V, **↓** flip H (see Orientation above).
- **Freeze (Space)** — Pause stream; snapshot and other keys still work.
- **Snapshot (S)** — Saves `~/Pictures/Infiray P2 Pro/snapshot_YYYYMMDD_HHMMSS.png` (current view with overlays).
- **Units (F)** — Toggle Celsius / Fahrenheit.
- **Alarm (A)** — Red border + "HOT!" when max temp ≥ 60°C.
- **Debug (D)** — Technical overlay (FPS, coords, temps, flags).
- **Help (I)** — Full keyboard reference.
- **Quit (Q)** — Exits; panel layout and settings are saved.

#### Saved between runs

Palette and invert, DYN on/off and gain, contrast mode and DDE parameters, DDE on/off, °C/°F, temperature-range
preset, rotation and flips, de-noise and IBP strength, RGB blend mode / alpha / hot threshold /
zoom, and the alarm threshold. Stored in
`~/Library/Application Support/Infiray P2 Pro/settings.json`; the file can be hand-edited, which
is currently the only way to change the alarm threshold (default 60 °C).
- **System tray** — Menu bar icon with Quit (macOS).

#### Virtual webcam (U) — for messengers / streaming

The **processed** thermal stream (after DDE, colormap, overlays) can be sent to a **virtual webcam** so Zoom, Teams, Telegram, OBS, etc. use it as the camera.

- **U** (or toolbar **VCAM**) — Toggle virtual cam output on/off.
- **macOS:** Install [OBS](https://obsproject.com/), then **Start Virtual Camera** in OBS (no need to add a source). The app will stream 640×480 into it; in your messenger choose **OBS Virtual Camera** as the camera.
- **Windows:** OBS Virtual Camera or compatible virtual cam; select it in the app.
- Requires `pyvirtualcam>=0.14` (installed in project venv). On macOS: OBS → Start Virtual Camera once, then close OBS; in the app press **U**. If the library or virtual cam is unavailable, the toggle will report an error and stay off.

---

### Files

| Item | Role |
|------|------|
| `app.py` | Main app: capture, pipeline, UI, panels, keyboard/mouse. |
| `fpn_correction.py` | Fixed-pattern noise: separable column/row/pixel model; flat-field (median + plane) and gated IRLMS scene-based NUC. |
| `denoise.py` | MC-Kalman temporal NR + edge-gated spatial mean. |
| `guided_upsample.py` | RGB-guided filter that transfers edges, not texture. |
| `burst_sr.py` | Float burst SR; HQ snapshot fusion + IBP + RL deconv. |
| `adaptive_sr.py` | Active live SR: bounded background joint reconstruction, phase coverage, motion/temperature masks, immediate current-frame fallback. |
| `test_adaptive_sr.py` | Detail recovery, sampling adjoint, subpixel registration, motion/occlusion and worker lifecycle regressions. |
| `requirements.txt` | Python dependencies. |
| `Infiray P2 Pro.spec` | PyInstaller spec for macOS .app bundle. |
| `~/Library/Application Support/Infiray P2 Pro/` | Saved panel layout and settings (created at runtime; outside the app bundle so they persist). |
| `make-demo-gif.sh` | Script to build `demo.gif` from `demo.mov` (ffmpeg + gifsicle). |

---

### Build (optional)

To build a standalone macOS app:

```bash
venv/bin/pyinstaller --noconfirm "Infiray P2 Pro.spec"
```

Output: `dist/Infiray P2 Pro.app`. The build includes ffmpeg and its shared libraries, along with the Python runtime and application dependencies. The packaged app prefers its bundled ffmpeg and can be copied to `/Applications` and launched from Finder without Homebrew on PATH. OBS Virtual Camera remains a separate installation for the optional virtual webcam output.

For development, `python app.py --sr --diagnostics` forces SR on and logs live FPS, accepted detail coverage and worker/foreground times every two seconds. Run imaging checks with `python -m unittest test_adaptive_sr test_imaging test_fpn_correction`.

The reconstruction currently uses a conservative approximate effective Gaussian PSF with regularization, motion rejection and bounded detail amplitude. It has synthetic validation, not a measured P2 Pro optical-resolution/temperature-accuracy guarantee. The live view does not wait for SR; actual FPS and useful detail depend on the camera, scene and host.

---

## License

Use and modify as needed. No warranty.
