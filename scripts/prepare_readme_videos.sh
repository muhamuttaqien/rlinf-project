#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/prepare_readme_videos.sh SUCCESS_MP4 FAILURE_MP4" >&2
  exit 2
fi

for src in "$1" "$2"; do
  [[ -f "$src" ]] || { echo "Video not found: $src" >&2; exit 1; }
done
command -v ffmpeg >/dev/null || { echo "Install ffmpeg first: sudo apt install ffmpeg" >&2; exit 1; }
mkdir -p docs/media
cp "$1" docs/media/success.mp4
cp "$2" docs/media/failure.mp4
# Small, looping visual previews for GitHub README. Full-length MP4 remains linked.
for kind in success failure; do
  ffmpeg -hide_banner -loglevel error -y -i "docs/media/${kind}.mp4" \
    -t 8 -vf 'fps=8,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer' \
    "docs/media/${kind}.gif"
done

echo "Prepared: docs/media/{success,failure}.{mp4,gif}"
echo "Review the GIFs and videos before publishing to verify labels."
echo "Stage media (even if *.mp4 is ignored): git add README.md scripts/prepare_readme_videos.sh docs/media/*.gif && git add -f docs/media/*.mp4"
