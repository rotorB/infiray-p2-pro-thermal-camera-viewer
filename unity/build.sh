#!/bin/bash
# Build the Quest APK from the command line, install it and start it.
# The Unity editor must be closed (one Unity instance per project).
#
# Usage: ./build.sh [adb-serial]    e.g. ./build.sh 192.168.4.42:5555
set -euo pipefail

cd "$(dirname "$0")"
UNITY=/Applications/Unity/Hub/Editor/6000.3.27f1/Unity.app/Contents/MacOS/Unity
PACKAGE=com.UnityTechnologies.com.unity.template.urpblank
LOG=Logs/build.log

# Thermal camera plugin (android/unityplugin): USB bridge + libp2prouvc + libuvc.
(cd ../android && ./gradlew -q :unityplugin:assembleRelease)
cp ../android/unityplugin/build/outputs/aar/unityplugin-release.aar Assets/Plugins/Android/p2pro-thermal.aar

# Run with the minimal PATH the editor gets when launched from the Dock: a
# shell PATH can put a broken /usr/local/bin/cmake ahead of the SDK's cmake,
# and Gradle then fails in configureCMake. Stop daemons that kept an old PATH.
pkill -f GradleDaemon || true
env -u ANDROID_HOME -u JAVA_HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  "$UNITY" -batchmode -quit -projectPath . -buildTarget Android \
  -executeMethod P2Pro.Editor.QuestBuild.Build -logFile "$LOG" \
  || { grep -E "error CS|What went wrong" -A2 "$LOG" | head -20; exit 1; }
grep "\[P2Pro\] Build" "$LOG"

ADB=(adb)
if [ $# -gt 0 ]; then ADB+=(-s "$1"); fi
"${ADB[@]}" install -r Builds/p2pro.apk
"${ADB[@]}" shell am start -n "$PACKAGE/com.unity3d.player.UnityPlayerGameActivity"
