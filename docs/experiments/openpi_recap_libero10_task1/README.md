# RECAP — LIBERO-10 Task 1

Independent continuation of the successful LIBERO-10 Task 0 experiment. The
**benchmark task index is 1**, the **LIBERO suite is `libero_10`**.
Never reuse Task 0 or LIBERO-Goal datasets, returns, advantages, or checkpoints.

## Baseline

Original π₀.₅ LIBERO-SFT policy: **7/30 (23.3%)** on Task 1, seeds 7–36.
This is a 30-episode estimate, not a claim that this task has identical
difficulty to Task 0.

## Design and differences from Task 0

| Item | Task 1 |
| --- | --- |
| Base policy | `/data1/Muttaqien/rlinf_data/checkpoints/torch/pi05_libero_sft_rlinf` |
| SFT | 30 **successful autonomous SFT rollouts**, not expert demonstrations |
| Rollouts | 1,024 SFT-policy episodes, successes + failures |
| Held-out | 64 SFT-policy episodes, successes + failures |
| Collector FPS | 10 (Task 0 original SFT dataset FPS was 20) |
| Critic | Success-only SFT split; 8,000 steps, LR 5e-5, global batch 256, warmup 500 |
| Advantages | Top 30% from rollout sidecar, gamma 1.0, 10-step lookahead |
| CFG policy | 3,000 steps, global batch 512, LR 1e-5, cosine scheduler with **30k-step schedule and 5k warmup**, positive CFG guidance scale 1.0 |
| Execution | GPU hardware rank 3 by default, after confirming availability |
| Dataset root | `/data1/Muttaqien/rlinf_data/recap_libero10_task1` |

**Methodological caveat:** Task 0's 30 SFT episodes were expert demonstrations;
Task 1's collector instead selects successful autonomous SFT-policy episodes.
Thus training configuration is similar, **but data sourcing is not identical**.
For exact replication, obtain the LIBERO-10 Task 1 expert demonstration data
and replace only the SFT dataset after confirming its format and prompt.
No data should be relabeled as expert when it is autonomous.

Task 0's policy training actually stopped at 3,000 steps while its configured
optimizer schedule was 30,000 steps with 5,000 warmup. These are retained
here to match effective behavior, not replaced by the Goal experiment's
3,000-step schedule and 500-step warmup.

Training checkpoints are large. Use `/data1` for outputs, NOT `/home`,
which previously ran out of disk space. Check GPU availability and NVIDIA NVML
library compatibility before training; do not restart shared Ray processes.

## 1. Pull and test

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project-sim
git pull origin branch-openpi-recap-sim
source ../rlinf-project/.venv/bin/activate
python -m unittest discover -s tests/unit_tests -p 'test_recap_libero10_task1.py' -v
nvidia-smi
```

## 2. Pilot collector into a separate dataset root

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib/nvml-fix:${LD_LIBRARY_PATH:-}"
CUDA_VISIBLE_DEVICES=3 python -m toolkits.recap_libero10_task1_collect \
  --output-root /data1/Muttaqien/rlinf_data/recap_libero10_task1_pilot \
  --sft-successes 2 --train-episodes 50 --eval-episodes 30
```

The pilot is intentionally isolated and is **not used for training**. Check the
generated manifests, episode counts, action/state shape, videos, and both
success and failure examples. A collection that stops mid-run is preserved
for inspection. Do not overwrite an incomplete dataset.

## 3. Full collector and preflight

```bash
CUDA_VISIBLE_DEVICES=3 python -m toolkits.recap_libero10_task1_collect \
  --sft-successes 30 --train-episodes 1024 --eval-episodes 64 \
  --max-sft-attempts 500

python -m toolkits.recap_libero10_task1_preflight --stage data
```

Collection uses disjoint seed ranges: SFT starts 40000, train 50000 and eval
60000. The manifest records benchmark Task 1. LeRobot's own task_index remains
0 because each collected dataset has a single instruction.

## 4. Compute returns + validate

```bash
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
  experiments/libero10_task1/recap_compute_returns
python -m toolkits.recap_libero10_task1_preflight --stage returns
```

## 5. Value critic (SIM-4)

```bash
export RLINF_NVIDIA_GPU_COUNT_OVERRIDE=4
bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh \
  experiments/libero10_task1/recap_value_model_sft \
  runner.logger.log_path=/data1/Muttaqien/rlinf_runs/value_sft/long_task1
```

The launcher creates a timestamped log directory by default but the explicit
logger override above redirects checkpoints to /data1. Confirm the final path
and stop at step 8,000. Validation is separate and should inform checkpoint
selection; a completed training loss alone does not demonstrate generalization.

## 6. Compute advantages + validate

Set the critic file path from a **Task 1 critic checkpoint**:

```bash
VALUE_CKPT=/absolute/path/to/TASK1_CRITIC/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt

CUDA_VISIBLE_DEVICES=3 bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh \
  experiments/libero10_task1/recap_compute_advantages --nproc 1 \
  "advantage.value_checkpoint=$VALUE_CKPT" \
  advantage.num_dataloader_workers_per_gpu=2 advantage.prefetch_factor=1

python -m toolkits.recap_libero10_task1_preflight --stage advantages
```

Verify the positive-label rate and continuous advantage distribution; these
labels belong to the Task 1 **rollout** dataset only.

## 7. Policy CFG training (SIM-6)

```bash
bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
  experiments/libero10_task1/cfg_rl_openpi \
  runner.logger.log_path=/data1/Muttaqien/rlinf_runs/cfg_rl/long_task1
```

Run inside tmux for long sessions. The output is isolated from Task 0/Goal;
verify all available GPUs and Ray placement before launching. Checkpoints at
steps 1,000, 2,000, 3,000.

## 8. Paired evaluation (SIM-7)

```bash
POLICY_CKPT=/absolute/path/to/TASK1_POLICY/checkpoints/global_step_3000
CUDA_VISIBLE_DEVICES=3 python -m toolkits.recap_libero_compare_videos \
  --checkpoint "$POLICY_CKPT" --suite libero_10 --task-index 1 \
  --episodes 30 --seed 7 --resume \
  --output /data1/Muttaqien/rlinf_runs/cfg_rl/recap_long_task1_30episodes
```

Then evaluate on fresh, disjoint seeds; do not choose the best checkpoint based
on the final test set. A 30-episode success difference is exploratory, not a
statistically conclusive improvement.
