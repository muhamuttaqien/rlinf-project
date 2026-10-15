#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export EMBODIED_PATH="$PWD/examples/embodiment"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
CKPT="$PWD/logs/20261015-00:30:14-maniskill_sac_flow_red_cube3/sac_flow_red_cube3/checkpoints/global_step_8000"
[[ -f "$CKPT/actor/model_state_dict/full_weights.pt" ]] || { echo "ERROR: missing actor weights at $CKPT"; exit 1; }
[[ -d "$CKPT/actor/dcp_checkpoint" ]] || { echo "ERROR: checkpoint missing actor/dcp_checkpoint"; exit 1; }
OUT="$PWD/logs/evaluation-red-cube3-failures-seed2000"
mkdir -p "$OUT"
export CKPT OUT
python - "$CKPT" <<'PY_VERIFY'
import sys, pathlib, torch
p = pathlib.Path(sys.argv[1])/'actor/model_state_dict/full_weights.pt'
try:
    # A structure check, not yet a policy load.
    obj=torch.load(p, map_location='cpu', weights_only=True, mmap=True)
    print(f'Checkpoint readable: {p} | top-level: {type(obj).__name__}')
    del obj
except Exception as exc:
    raise SystemExit(f'Checkpoint readability test failed: {exc}')
PY_VERIFY
python "$EMBODIED_PATH/evaluate_red_cube3.py" \
  --config-path "$EMBODIED_PATH/config" \
  --config-name maniskill_sac_flow_red_cube3 \
  "+runner.resume_dir=$CKPT" \
  "runner.logger.log_path=$OUT" \
  "env.eval.seed=2000" \
  "env.eval.total_num_envs=20" \
  "env.eval.rollout_epoch=5" \
  "env.eval.auto_reset=false" \
  "env.eval.ignore_terminations=true" \
  "env.eval.use_fixed_reset_state_ids=false" \
  "env.eval.max_steps_per_rollout_epoch=100" \
  "env.eval.video_cfg.save_video=true" \
  "env.eval.video_cfg.video_base_dir=$OUT/video" \
  2>&1 | tee "$OUT/evaluation.log"
echo "Evaluation output: $OUT"

python summarize_red_cube_failures.py
