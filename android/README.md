# P2 Pro Thermal for Meta Quest 3

Android (Horizon OS) app that reads the Infiray P2 Pro over the headset's USB-C port and shows
the live thermal image in a panel, decoded the same way as the desktop viewer (`app.py`):
YUYV 256×384 frame, bottom 192 rows = uint16 per pixel, `°C = raw / 64 − 273.15`.

## Build and install

Requires JDK 17 and the Android SDK (platform 35, build-tools 35).

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
echo "sdk.dir=/opt/homebrew/share/android-commandlinetools" > local.properties
./gradlew assembleDebug
adb install -r -g app/build/outputs/apk/debug/app-debug.apk
```

The headset has one USB-C port, so with the camera plugged in use ADB over Wi-Fi:
`adb tcpip 5555` (while on the cable), then `adb connect <headset-ip>:5555`.

## Using it

1. Plug the P2 Pro into the headset. Horizon OS offers to open P2 Pro Thermal.
2. Allow camera access and USB access when asked (once).
3. Point the controller at the image to read the temperature under the pointer.

Buttons: **Palette** (Inferno, Turbo, Ironbow, White hot, Black hot), **Range** (Auto, Room
15–35 °C, Wide −20–150 °C), **Rotate** (90° steps), **Mirror**.

The image shows min (blue) and max (red) markers, the center spot temperature, a color bar
with the current range, and FPS.

## Horizon OS specifics

- UVC devices need `horizonos.permission.USB_CAMERA` (dangerous, runtime) in addition to
  `android.permission.CAMERA`; without it `UsbManager.hasPermission` stays false.
- The kernel `uvcvideo` driver binds the camera (`/dev/video*`, not readable by apps) and the
  external camera is not published to Camera2, so the app detaches the kernel driver with a
  forced `claimInterface` and reads the stream through libuvc
  ([UVCAndroid](https://github.com/shiyinghan/UVCAndroid)).
- Camera2 would convert YUYV to YUV_420_888 anyway, which destroys the 16-bit temperature data.
