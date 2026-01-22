#!/usr/bin/env bash
set -euo pipefail

# Settings
INPUT_DIR="./projections"
PATTERN="projection_%04d.png"
FPS=60
OUTPUT="output.mp4"

echo "Creating MP4 (H.264, high quality)..."
ffmpeg -hide_banner -y \
  -framerate "${FPS}" -start_number 0 \
  -i "${INPUT_DIR}/${PATTERN}" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -preset slow \
  -crf 16 \
  -movflags +faststart \
  -r "${FPS}" \
  "${OUTPUT}"

echo "Done → ${OUTPUT}"
