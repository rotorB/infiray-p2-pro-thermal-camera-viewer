# Neural upscale models (optional)

To enable **neural super-resolution** (key **Y**), install `opencv-contrib-python`. A LapSRN x2 model is included; the app also supports `LapSRN_x3.pb` if you add it.

## 1. Install OpenCV contrib

```bash
pip uninstall opencv-python  # if present, to avoid conflict
pip install opencv-contrib-python
```

Or use the optional requirements file from the project root:

```bash
pip install -r requirements-superres.txt
```

## 2. Models

- **LapSRN_x2.pb** — included (from [TF-LapSRN](https://github.com/fannymonori/TF-LapSRN)); app uses it for 2× then resizes to 768×576.
- **LapSRN_x3.pb** — optional; if you add it here, the app uses it for direct 3× (no extra resize). Download from the same repo or [learnopencv Super Resolution](https://learnopencv.com/super-resolution-in-opencv/).

The app loads the first available model (x3 then x2). With `opencv-contrib-python` installed, key **Y** toggles neural upscale.
