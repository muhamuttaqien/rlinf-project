#!/usr/bin/env bash
# Source from workstation launchers after REPO_PATH is set.
export RLINF_DATA_ROOT="${RLINF_DATA_ROOT:-/data1/Muttaqien}"

if [[ ! -d "$RLINF_DATA_ROOT" || ! -w "$RLINF_DATA_ROOT" ]]; then
    printf 'ERROR: RLINF_DATA_ROOT must exist and be writable: %s\n' "$RLINF_DATA_ROOT" >&2
    printf 'Mount/create the workstation storage first or set RLINF_DATA_ROOT explicitly.\n' >&2
    return 1
fi

export HF_HOME="${HF_HOME:-$RLINF_DATA_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$RLINF_DATA_ROOT/cache/transformers}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$RLINF_DATA_ROOT/cache/openpi}"
export TORCH_HOME="${TORCH_HOME:-$RLINF_DATA_ROOT/cache/torch}"
export WANDB_DIR="${WANDB_DIR:-$RLINF_DATA_ROOT/logs/wandb}"
export TMPDIR="${TMPDIR:-$RLINF_DATA_ROOT/tmp}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
    "$TRANSFORMERS_CACHE" "$OPENPI_DATA_HOME" "$TORCH_HOME" "$WANDB_DIR" "$TMPDIR"
