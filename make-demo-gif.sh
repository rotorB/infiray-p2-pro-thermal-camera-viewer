#!/bin/bash
# Speed multiplier for output video (2 = 2x faster, 3 = 3x). More aggressive mpdecimate = drop more similar frames.
SPEED=2
# Step 1: MP4 — no audio, drop similar frames, then play at SPEED x (shorter video)
ffmpeg -i demo.mov -vf "mpdecimate=frac=0.33,setpts=N/(FRAME_RATE*${SPEED})/TB,scale=600:-1" -c:v libx264 -r 60 -an -y demo_600.mp4

# Step 2: Generate palette from the decimated video
ffmpeg -i demo_600.mp4 -vf "scale=600:-1:flags=lanczos,palettegen=stats_mode=diff:max_colors=128" -update 1 -y palette.png

# Step 3: Apply palette; [1:v] = palette from second input
ffmpeg -i demo_600.mp4 -i palette.png -lavfi "scale=600:-1:flags=lanczos[f];[f][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=1" -r 10 -y demo.gif

# Optional: compress with gifsicle; --delay=3 (30ms per frame) = faster playback like original
if command -v gifsicle &>/dev/null; then
  gifsicle -O3 --lossy=80 --delay=3 -o demo.gif demo.gif
fi

# Cleanup
rm -f palette.png

echo Done: demo.gif
