#!/bin/bash
# Generate palette (single file: -update 1)
ffmpeg -i demo.mov -vf "scale=400:-1:flags=lanczos,palettegen=stats_mode=diff:max_colors=128" -update 1 -y palette.png

# Apply palette; [1:v] = palette from second input
ffmpeg -i demo.mov -i palette.png -lavfi "scale=400:-1:flags=lanczos[f];[f][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=1" -r 10 -y demo.gif

# Optional: compress with gifsicle; --delay=3 (30ms per frame) = faster playback like original
if command -v gifsicle &>/dev/null; then
  gifsicle -O3 --lossy=80 --delay=3 -o demo.gif demo.gif
fi

# Cleanup
rm -f palette.png

echo Done: demo.gif
