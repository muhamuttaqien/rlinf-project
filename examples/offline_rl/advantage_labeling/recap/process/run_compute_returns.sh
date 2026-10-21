#!/bin/bash
# Compute returns for LeRobot datasets
#
# Usage:
#   bash run_compute_returns.sh [CONFIG_NAME] [EXTRA_ARGS]
#
# Examples:
#   bash run_compute_returns.sh
#   bash run_compute_returns.sh compute_returns data.data_root=/path/to/data
#   bash run_compute_returns.sh compute_returns data.dataset_path=/path/to/dataset data.dataset_type=sft

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"
set -e

source switch_env openpi 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH=$(dirname $(dirname $(dirname $(dirname $(dirname "$SCRIPT_DIR")))))
export OFFLINE_RL_CONFIG="${REPO_PATH}/examples/offline_rl/config"
export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH
cd "$SCRIPT_DIR"

CONFIG_NAME=${1:-"recap_compute_returns"}
shift 1 2>/dev/null || true

# Preserve each Hydra override as a single argument. Building a command
# string and evaluating it causes Bash brace expansion to split dictionary
# overrides such as 'data.train_data_paths=[{dataset_path:/tmp/x,type:sft}]'.
CMD=(python compute_returns.py --config-path "$OFFLINE_RL_CONFIG" --config-name "$CONFIG_NAME")
CMD+=("$@")

echo "=========================================="
echo "Return Computation for LeRobot Datasets"
echo "=========================================="
echo "  Config: $CONFIG_NAME"
if [ "$#" -gt 0 ]; then
    printf '  Extra args: '
    printf '%q ' "$@"
    printf '\n'
fi
echo ""

printf 'Command: '
printf '%q ' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}"

echo ""
echo "=========================================="
echo "Return computation complete!"
echo "=========================================="
