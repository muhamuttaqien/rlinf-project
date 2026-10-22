#!/usr/bin/env bash
# Move generated RLinf/StackCube data to /data1/Muttaqien without changing
# existing config paths. Existing /data1/Muttaqien/openpi_checkpoints is untouched.
# This is a one-time operation. Stop training/evaluation before using --apply.
set -euo pipefail

if [[ $# -ne 1 ]] || [[ "$1" != "--plan" && "$1" != "--apply" ]]; then
  echo "Usage: bash scripts/migrate_stackcube_data_to_data1.sh --plan|--apply"
  exit 2
fi

mode="$1"
project="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
base="/data1/Muttaqien"
store="$base/rlinf_data"

if [[ ! -d "$base" || ! -w "$base" ]]; then
  echo "ERROR: $base does not exist or is not writable."
  exit 1
fi
if [[ "$mode" == "--apply" ]] && ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: rsync is required; no files have been changed."
  exit 1
fi

hf_home="$(printenv HF_HOME || true)"
if [[ -z "$hf_home" ]]; then
  hf_home="$HOME/.cache/huggingface"
fi
lerobot_home="$(printenv HF_LEROBOT_HOME || true)"
if [[ -z "$lerobot_home" ]]; then
  lerobot_home="$hf_home/lerobot"
fi
hub_home="$(printenv HF_HUB_CACHE || true)"
if [[ -z "$hub_home" ]]; then
  hub_home="$hf_home/hub"
fi

if [[ "$mode" == "--apply" ]]; then
  # Never begin a migration onto an already nearly-full destination.
  free_bytes="$(df -PB1 "$base" | awk 'NR==2 {print $4}')"
  if (( free_bytes < 15 * 1024 * 1024 * 1024 )); then
    echo "ERROR: /data1 has less than 15 GiB free; preserve originals."
    exit 1
  fi
  mkdir -p -- "$store"
fi

migrate_one() {
  local src="$1"
  local dst="$2"
  local remaining

  if [[ -L "$src" ]]; then
    if [[ "$(readlink -f -- "$src")" == "$(realpath -m -- "$dst")" ]]; then
      echo "ALREADY MOVED: $src"
    else
      echo "SKIP (existing symlink): $src"
    fi
    return 0
  fi
  if [[ ! -d "$src" ]]; then
    echo "SKIP (not found): $src"
    return 0
  fi
  if [[ -L "$dst" || ( -e "$dst" && ! -d "$dst" ) ]]; then
    echo "ERROR: destination is not a regular directory: $dst"
    return 1
  fi

  if [[ "$mode" == "--plan" ]]; then
    echo "MOVE: $(du -sh -- "$src" | cut -f1) | $src -> $dst"
    return 0
  fi

  echo "COPY: $src -> $dst"
  mkdir -p -- "$dst"
  # Resumable copy that includes dotfiles, ownership, times and symlinks.
  rsync -aH -- "$src/" "$dst/"

  echo "VERIFY (checksums): $src"
  # Compare contents and metadata without modifying either tree. Never
  # remove the original unless the copy fully passes this verification.
  if ! remaining="$(rsync -aHcni --out-format='%n' -- "$src/" "$dst/")"; then
    echo "ERROR: verification failed; original retained: $src"
    return 1
  fi
  if [[ -n "$remaining" ]]; then
    echo "ERROR: verification found differences; original retained: $src"
    return 1
  fi

  rm -rf --one-file-system -- "$src"
  ln -s -- "$dst" "$src"
  echo "MOVED AND LINKED: $src"
}

echo "Mode: $mode"
echo "Destination: $store"
df -h -- "$base" | tail -n 1

# Preserve the source repo and .venv; migrate data/storage, not executable code.
migrate_one "$project/../results" "$store/results"
migrate_one "$project/checkpoints" "$store/checkpoints"
migrate_one "$project/assets/pi05_stackcube_panda_joint" "$store/pi05_stackcube_panda_joint_assets"
migrate_one "$lerobot_home" "$store/lerobot"
migrate_one "$hub_home" "$store/huggingface_hub"

if [[ "$mode" == "--apply" ]]; then
  mkdir -p -- "$store/ray"
  echo "Migration finished. The original paths now point into /data1."
  echo "For future local Ray sessions:"
  echo "  export RAY_TMPDIR=\"$store/ray\""
  echo "Preserved: $base/openpi_checkpoints"
  df -h -- "$base" "$HOME" | tail -n 2
else
  echo "No files changed. Stop training/evaluation, then run with --apply."
fi
