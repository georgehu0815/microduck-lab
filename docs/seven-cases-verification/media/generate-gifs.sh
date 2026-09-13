#!/usr/bin/env bash
set -euo pipefail

media_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$media_dir/../../.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

make_gif() {
  local name="$1"
  local start="$2"
  local source="$3"
  local palette="$tmp_dir/$name.png"
  local output="$media_dir/$name.gif"
  local filters="fps=6,scale=360:-1:flags=lanczos"

  ffmpeg -v error -y -ss "$start" -t 4 -i "$repo_root/$source" \
    -vf "$filters,palettegen=stats_mode=diff" "$palette"
  ffmpeg -v error -y -ss "$start" -t 4 -i "$repo_root/$source" -i "$palette" \
    -lavfi "$filters [x]; [x][1:v] paletteuse=dither=sierra2_4a" "$output"
}

make_gif dance 2 \
  "rlx/runs/studio/dance/dance-e2e-20260907-low-noise/render/ep0.mp4"
make_gif swing 13 \
  "rlx/runs/studio/swing/swing-e2e-20260907-v3/render/ep0.mp4"
make_gif running 2 \
  "rlx/runs/studio/running/running-e2e-20260907-v4/render/ep0.mp4"
make_gif stilts 2 \
  "rlx/runs/studio/stilts/stilts-e2e-20260907-v3/render/ep0.mp4"
make_gif backflip 0 \
  "rlx/runs/studio/backflip/backflip-e2e-20260908-v5/render/ep0.mp4"
make_gif basketball 24 \
  "rlx/runs/studio/basketball/basketball-balance-01/render/rollout.mp4"
make_gif bridge 13 \
  "rlx/runs/studio/bridge/bridge-studio-02/render/ep0.mp4"
