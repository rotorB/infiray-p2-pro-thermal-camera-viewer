# P2 Pro Thermal Camera Viewer

![Demo](demo_400.gif)

Real-time thermal imaging desktop app for the **Infiray P2 Pro** USB camera. Captures the raw YUYV stream (256×384), extracts the 256×192 thermal layer, converts to temperature, and displays with colormaps, measurement tools, and optional analysis overlays. Optimized for **macOS** (AVFoundation + ffmpeg); includes system tray and optional PyInstaller build.

---

## Requirements

- **Hardware:** Infiray P2 Pro (or compatible USB thermal camera: PureThermal, "USB Camera" with same protocol).
- **System:** macOS (AVFoundation). **ffmpeg** must be installed (e.g. `brew install ffmpeg`).
- **Python:** 3.8+. Dependencies: `opencv-python`, `numpy`, `pystray`, `Pillow`, `pyvirtualcam` (optional, for virtual webcam).

---

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Connect the P2 Pro; the app auto-detects it and opens the viewer. Use **I** in-app for the full control list.

---

## Functionality Overview

### Core imaging

- **Live thermal stream** — 256×192 thermal data, 3× upscale to 768×576, 25 FPS.
- **Temperature** — Raw 16-bit → Celsius (formula: `/64 - 273.15`), clip -20…150°C for display.
- **Range presets (0):** Auto (frame min–max), Room (15–35°C), Wide (-20–150°C).
- **Palettes (P, 1–9):** Inferno, Jet, Hot, Plasma, Magma, Viridis, Rainbow, Bone, Ocean.

### Image quality

- **DDE – High Quality (H):** CLAHE + Lanczos upscale + unsharp mask. **[ / ]** = contrast, **-** **=** = sharpness.
- **FPN correction:** **C** = running per-pixel calibration; **X** = one-frame reference (capture now, subtract offset; **X** again = reset).

### Orientation (rotate & flip)

- **Left / Right arrows** — Rotate image by 90° (0°, 90°, 180°, 270°).
- **Up arrow** — Flip vertically (mirror top–bottom).
- **Down arrow** — Flip horizontally (mirror left–right).

Useful when the camera is mounted upside-down or sideways; all overlays and measurements follow the transformed view.

### Measurements

- **Cursor** — Green crosshair + temperature at mouse position.
- **Min/Max markers (M)** — Blue (min) and red (max) triangles with values.
- **ROI (R)** — Draw rectangle; shows min / avg / max for the region.
- **Line profile (L)** — Draw a line on the image; panel shows temperature graph along the line (draggable/resizable panel).
- **Delta-T (V)** — Capture reference frame; display shows temperature difference (diverging colormap). Cursor shows absolute + delta. **V** again = reset.

### Analysis overlays

- **Isotherms (T)** — Contour lines at 6 temperature levels with labels.
- **Histogram (B)** — Temperature distribution in a draggable/resizable panel.
- **Trend (W)** — Cursor temperature over time (last 300 samples) in a draggable/resizable panel.
- **Anomaly detection (N)** — Z-score (mean + 2σ); highlights hot regions and shows count + threshold.

### Panels (Trend, Histogram, Line profile)

- **Drag** by the title bar; **resize** by the bottom-right grip.
- Layout is saved to `.p2pro_layout.json` and restored on next run.
- White border indicates drag state.

### Other

- **Rotate & flip** — **←/→** rotate 90°; **↑** flip V, **↓** flip H (see Orientation above).
- **Freeze (Space)** — Pause stream; snapshot and other keys still work.
- **Snapshot (S)** — Saves `snapshot_YYYYMMDD_HHMMSS.png` (current view with overlays).
- **Units (F)** — Toggle Celsius / Fahrenheit.
- **Alarm (A)** — Red border + "HOT!" when max temp ≥ 60°C.
- **Debug (D)** — Technical overlay (FPS, coords, temps, flags).
- **Help (I)** — Full keyboard reference.
- **Quit (Q)** — Exits; panel layout is saved.
- **System tray** — Menu bar icon with Quit (macOS).

### Virtual webcam (U) — for messengers / streaming

The **processed** thermal stream (after DDE, colormap, overlays) can be sent to a **virtual webcam** so Zoom, Teams, Telegram, OBS, etc. use it as the camera.

- **U** (or toolbar **VCAM**) — Toggle virtual cam output on/off.
- **macOS:** Install [OBS](https://obsproject.com/), then **Start Virtual Camera** in OBS (no need to add a source). The app will stream 640×480 into it; in your messenger choose **OBS Virtual Camera** as the camera.
- **Windows:** OBS Virtual Camera or compatible virtual cam; select it in the app.
- Requires `pip install pyvirtualcam`. If the library or OBS virtual cam is unavailable, the toggle will report an error and stay off.

---

## Project layout

| Item | Role |
|------|------|
| `app.py` | Main app: capture, pipeline, UI, panels, keyboard/mouse. |
| `fpn_correction.py` | Fixed-pattern noise: per-pixel running mean, offset subtraction. |
| `requirements.txt` | Python dependencies. |
| `Infiray P2 Pro.spec` | PyInstaller spec for macOS .app bundle. |
| `.p2pro_layout.json` | Saved panel positions/sizes (created at runtime). |
| `make-demo-gif.sh` | Script to build `demo.gif` from `demo.mov` (ffmpeg + gifsicle). |

---

## Build (optional)

To build a standalone macOS app:

```bash
pyinstaller "Infiray P2 Pro.spec"
```

Output: `dist/Infiray P2 Pro.app`. Ensure ffmpeg is on the system PATH (or bundle it in the spec) for the packaged app to find the camera.

---

## License

Use and modify as needed. No warranty.
