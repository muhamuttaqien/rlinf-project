#!/usr/bin/env bash
# Run Clean V1 Experiment B from the original pi05_base. No resume, no overwrite.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${PI05_MODEL_PATH:?Export PI05_MODEL_PATH to the ORIGINAL pretrained pi05_base directory}"
if [[ ! -f "$PI05_MODEL_PATH/model.safetensors" ]]; then
  echo "ERROR: original pi05_base/model.safetensors not found: $PI05_MODEL_PATH" >&2
  exit 1
fi
for candidate in \
  "$PI05_MODEL_PATH/full_weights.pt" \
  "$PI05_MODEL_PATH/model_state_dict/full_weights.pt" \
  "$PI05_MODEL_PATH/actor/model_state_dict/full_weights.pt"; do
  if [[ -f "$candidate" ]]; then
    echo "ERROR: model path contains a trained full_weights.pt: $candidate" >&2
    exit 1
  fi
done

export HF_LEROBOT_HOME=/data1/Muttaqien/rlinf_data/lerobot
export STACKCUBE_DATASET=local/stackcube_pi05_clean_v1
export STACKCUBE_NORM_STATS=/data1/Muttaqien/rlinf_data/pi05_stackcube_clean_v1_assets/local/stackcube_pi05_clean_v1/norm_stats.json
export STACKCUBE_RESULTS_ROOT=/data1/Muttaqien/rlinf_data/results
export EMBODIED_PATH="$REPO_ROOT/examples/sft"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export RAY_ADDRESS=local
export RAY_TMPDIR=/data1/Muttaqien/rt
export RLINF_RAY_DISABLE_DASHBOARD=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RLINF_RAY_LOCAL_NUM_GPUS=1

if [[ ! -f "$STACKCUBE_NORM_STATS" ]]; then
  echo "ERROR: Clean V1 norm stats missing: $STACKCUBE_NORM_STATS" >&2
  exit 1
fi
if [[ ! -f "$HF_LEROBOT_HOME/$STACKCUBE_DATASET/meta/info.json" ]]; then
  echo "ERROR: Clean V1 LeRobot dataset missing" >&2
  exit 1
fi

RUN_ROOT="$STACKCUBE_RESULTS_ROOT/maniskill_stackcube_pi05_clean_v1_1800_32d"
if [[ -e "$RUN_ROOT" ]]; then
  echo "ERROR: refusing to reuse Experiment B results: $RUN_ROOT" >&2
  exit 1
fi

mkdir -p "$RAY_TMPDIR"
echo "EXPERIMENT B: original pretrained base=$PI05_MODEL_PATH"
echo "EXPERIMENT B: dataset=$STACKCUBE_DATASET steps=1800 loss=32D"
echo "EXPERIMENT B: results=$RUN_ROOT"
PYTHONPATH="$REPO_ROOT" python examples/sft/train_vla_sft.py \
  --config-path "$REPO_ROOT/examples/sft/config" \
  --config-name maniskill_stackcube_panda_sft_openpi_pi05_clean_v1_1800_32d
