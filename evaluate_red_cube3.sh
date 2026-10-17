#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
export EMBODIED_PATH="$PWD/examples/embodiment"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"

CKPT="$PWD/logs/20261016-19:07:23-maniskill_sac_flow_red_cube3/sac-red-cube/checkpoints/global_step_8000"
[[ -f "$CKPT/actor/model_state_dict/full_weights.pt" ]] || { echo "ERROR: missing actor weights at $CKPT"; exit 1; }
[[ -d "$CKPT/actor/dcp_checkpoint" ]] || { echo "ERROR: checkpoint missing actor/dcp_checkpoint"; exit 1; }

python - "$CKPT" <<'PY_VERIFY'
import pathlib, sys, torch
p = pathlib.Path(sys.argv[1]) / 'actor/model_state_dict/full_weights.pt'
try:
    obj = torch.load(p, map_location='cpu', weights_only=True, mmap=True)
    print(f'Checkpoint readable: {p} | top-level: {type(obj).__name__}')
    del obj
except Exception as exc:
    raise SystemExit(f'Checkpoint readability test failed: {exc}')
PY_VERIFY

# Set SAVE_VIDEO=false to avoid generating evaluation videos when disk space is tight.
SAVE_VIDEO="${SAVE_VIDEO:-true}"
for SEED in 1000 2000 3000 4000 5000; do
  OUT="$PWD/logs/evaluation-red-cube3-seed${SEED}"
  mkdir -p "$OUT"
  echo "===== Evaluating seed ${SEED}: 20 environments x 5 batches = 100 episodes ====="

  python "$EMBODIED_PATH/evaluate_red_cube3.py" \
    --config-path "$EMBODIED_PATH/config" \
    --config-name maniskill_sac_flow_red_cube3 \
    "+runner.resume_dir=$CKPT" \
    "runner.logger.log_path=$OUT" \
    "runner.logger.project_name=rlinf-project" \
    "runner.logger.experiment_name=sac-red-cube-eval-seed${SEED}" \
    "runner.logger.logger_backends=[tensorboard,wandb]" \
    "env.eval.seed=$SEED" \
    "env.eval.total_num_envs=20" \
    "env.eval.rollout_epoch=5" \
    "env.eval.auto_reset=false" \
    "env.eval.ignore_terminations=true" \
    "env.eval.use_fixed_reset_state_ids=false" \
    "env.eval.max_steps_per_rollout_epoch=100" \
    "env.eval.video_cfg.save_video=$SAVE_VIDEO" \
    "env.eval.video_cfg.video_base_dir=$OUT/video" \
    2>&1 | tee "$OUT/evaluation.log"

  python - "$OUT/eval_metrics.json" "$SEED" <<'PY_RESULT'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); seed = sys.argv[2]
if not p.exists():
    raise SystemExit(f'ERROR: metrics missing for seed {seed}: {p}')
m = json.loads(p.read_text())
print(f"SEED {seed}: episodes={int(m['num_trajectories'])}, "
      f"success_once={100*m['success_once']:.1f}%, "
      f"success_at_end={100*m['success_at_end']:.1f}%")
PY_RESULT
done

python - "$PWD/logs" <<'PY_SUMMARY'
import json, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
rows = []
for seed in (1000, 2000, 3000, 4000, 5000):
    path = root / f'evaluation-red-cube3-seed{seed}' / 'eval_metrics.json'
    m = json.loads(path.read_text())
    rows.append((seed, int(m['num_trajectories']), float(m['success_once']), float(m['success_at_end'])))
print('\nSEED  EPISODES  SUCCESS ONCE  SUCCESS AT END')
for seed, n, once, end in rows:
    print(f'{seed:4d}  {n:8d}  {once:11.1%}  {end:14.1%}')
total = sum(r[1] for r in rows)
once = sum(r[1] * r[2] for r in rows) / total
end = sum(r[1] * r[3] for r in rows) / total
print(f'ALL   {total:8d}  {once:11.1%}  {end:14.1%}')
print('Seed-to-seed std (sample): ' +
      f'success_once={statistics.stdev(100*r[2] for r in rows):.2f} pp, ' +
      f'success_at_end={statistics.stdev(100*r[3] for r in rows):.2f} pp')
PY_SUMMARY
