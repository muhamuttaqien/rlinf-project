# Task-Specific RECAP — LIBERO-Goal Task 0

This experiment **trains a new value critic and RECAP policy for Goal Task 0**.
It is separate from the completed LIBERO-10 Task 0 RECAP checkpoint and the
cross-suite transfer evaluation of that checkpoint.

**Status (2026-11-13):** Goal Task 0 data collection and return computation
passed preflight (48 SFT successes; 200 mixed train; 40 mixed held-out eval).
The original success-only critic completed 8,000 steps but showed poor
held-out validation. Separate mixed-outcome pilot/full configs are prepared
but have **not yet been run**. SIM-5/6/7 are pending.

## Experiment design

| Item | Setting |
| --- | --- |
| Benchmark | LIBERO-Goal, task index 0 |
| Initial policy | Existing `pi05_libero_sft_rlinf` SFT checkpoint |
| SFT data | 48 *successful autonomous SFT rollouts* (not expert demos) |
| Mixed training rollouts | 200 baseline rollouts, successes and failures |
| Held-out evaluation rollouts for the value model | 40 independently seeded baseline rollouts |
| SIM-4 value critic | 8,000 optimizer steps |
| SIM-5 labeling | Goal-task-specific return sidecars and advantages |
| SIM-6 RECAP fine-tuning | 3,000 optimizer steps, from the ORIGINAL SFT checkpoint |
| Comparison after training | 20 matched LIBERO-Goal Task 0 episodes per policy |

The collector is a **proof-of-concept** replacement for a task-specific
expert-demonstration dataset. Autonomous successful rollouts are *not* the same
as expert teleoperation demonstrations. Review task success balance and value
validation before making paper-level claims.

The existing 20-episode Goal transfer gallery contains MP4s, wrist MP4s and
sampled states, but **not all executed actions**. Do **not** use it as RECAP
training data and do not relabel LIBERO-10 data as Goal Task 0.

## GPU detection workaround: Ray/NVML shows 0, PyTorch shows 4

On workstation `pca03395`, `nvidia-smi -L` and `torch.cuda.device_count()`
reported four GPUs, but Ray's NVIDIA manager returned 0 because NVML
initialization failed. RLinf's FSDP worker consequently received
`accelerator_type=NO_ACCEL` and failed **before any training**.

The branch provides an **opt-in, NVIDIA-only** workaround through
`RLINF_NVIDIA_GPU_COUNT_OVERRIDE`. This changes only RLinf's own NVIDIA
enumeration when Ray's lower-level detector returns 0. It does **not**
change Ray's resource scheduler. Both layers must agree.

1. Check that no other Ray cluster/job is active on the workstation.
   Never `ray stop` a shared cluster without coordinating with its users.
2. Confirm `nvidia-smi -L` and `torch.cuda.device_count()` each show 4.
3. In the SAME shell, **before starting Ray**, run:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RLINF_NVIDIA_GPU_COUNT_OVERRIDE=4

python -m unittest discover -s tests/unit_tests -p "test_recap_libero_ray_gpu_fallback.py" -v

# ONLY if no Ray instance is already running:
ray start --head --num-gpus=4

ray status
```

4. Check that Ray advertises four GPU resources and that a Ray worker
   inherited the explicit RLinf override:

```bash
python - <<'PY'
import ray
from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager
from ray._private.accelerators.nvidia_gpu import NvidiaGPUAcceleratorManager

ray.init(address="auto")
print("Ray resources:", ray.cluster_resources())
print("Native Ray NVML detector:", NvidiaGPUAcceleratorManager.get_current_node_num_accelerators())
print("Driver RLinf detector:", NvidiaGPUManager.get_num_devices())

@ray.remote
def worker_probe():
    import os
    from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager
    return {
        "override": os.environ.get("RLINF_NVIDIA_GPU_COUNT_OVERRIDE"),
        "rlinf_gpus": NvidiaGPUManager.get_num_devices(),
    }

print("Ray worker:", ray.get(worker_probe.remote()))
ray.shutdown()
PY
```

Expected: Ray resources include `'GPU': 4.0`, the RLinf driver detector
reports 4, and the Ray worker reports `override=4` and `rlinf_gpus=4`.
The *native* Ray NVML detector can still report 0: that's why the explicit
workaround exists. Do not proceed if these checks disagree.

5. In the same shell, launch the revised mixed-outcome pilot (Step 3):

```bash
bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh \
  experiments/libero_goal_task0/recap_value_model_mixed_pilot
```

The original success-only config places workers on GPU rank 0, while the
new mixed-outcome pilot/full configs use rank 2. Confirm the RLinf startup
log reports the intended GPU-assigned placement before trusting a run.

If NVML works again, simply `unset RLINF_NVIDIA_GPU_COUNT_OVERRIDE` and
start Ray normally (with GPU resources verified).

## Run the pipeline

From a clean `branch-openpi-recap-sim` checkout:

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project-sim
git pull origin branch-openpi-recap-sim
source ../rlinf-project/.venv/bin/activate
python -m unittest discover -s tests/unit_tests -p "test_recap_libero_goal_task0.py" -v
```

### Step 1 — Collect Goal-specific data

```bash
python -m toolkits.recap_libero_goal_collect
python -m toolkits.recap_libero_goal_preflight
```

Default output (separate from LIBERO-10):

```text
/data1/Muttaqien/rlinf_data/recap_libero_goal_task0/
    libero_goal_task0_sft/
    libero_goal_task0_train/
    libero_goal_task0_eval/
```

The collector runs the original SFT policy on LIBERO-Goal Task 0 and writes
**actual executed actions**, RGB images, wrist images, 8-D states, 7-D actions,
task prompts, episode-success flags, and episode-end markers into LeRobot
datasets. Separate seed ranges are used for SFT/train/eval. Output directories
are never overwritten; complete matching splits are reused.

You can collect one split at a time:

```bash
python -m toolkits.recap_libero_goal_collect --split sft
python -m toolkits.recap_libero_goal_collect --split train
python -m toolkits.recap_libero_goal_collect --split eval
```

If a split is incomplete, **inspect the partial dataset** before choosing a
new `--output-root`; do not blindly delete data or let two collectors write
to one directory.

### Step 2 — Compute returns (prerequisite to the value model)

```bash
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
  experiments/libero_goal_task0/recap_compute_returns

python -m toolkits.recap_libero_goal_preflight --stage returns
```

The new tag `libero_goal_task0_recap` is used for all three return sidecars.
Failed episodes carry the configured terminal failure penalty.

### Step 3 — SIM-4: Mixed-outcome value critic (revised)

**Why:** The original success-only critic completed 8,000 steps but evaluated
poorly on held-out mixed rollouts (training MAE ~0, held-out MAE 0.348;
held-out loss 8.51, categorical best accuracy 8.4%). Its training split
contained only the 48 successful SFT rollouts, while evaluation contained
failures whose normalized returns could fall below -1.

**Preservation:** The original config
`recap_value_model_sft.yaml` and its completed checkpoint are retained
as the **success-only reference**. Never overwrite or mistake that checkpoint
for the mixed-outcome critic.

The **two new configs** have their own experiment/log directories:

| Config | Training split | Held-out eval | Budget | Evaluate | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| `recap_value_model_mixed_pilot` | 200 mixed train (163 success, 37 failure) | 40 eval (33 success, 7 failure) | 1,000 steps | every 250 | every 500 |
| `recap_value_model_mixed_full` | Same mixed train | Same eval | 8,000 steps | every 500 | every 500 |

Both preserve the original critic architecture, batch sizes (32/256),
optimizer LR (5e-5), value LR (1e-4), BF16, and **500-step LR warmup**.
The pilot retains the full 8,000-step optimizer schedule and *stops early*
at 1,000 for diagnosis. Both select GPU rank 2 through
`cluster.component_placement.actor,env,rollout: "2"`, following the
workstation's earlier GPU-1 watchdog error.

The ValueSFT worker reads `return_min` and `return_max` from
**training datasets only** if no explicit range is configured. Because each
mixed config names only `libero_goal_task0_train` in `train_data_paths`,
the normalization now derives from that dataset's
`meta/stats.json` and is applied consistently to the held-out eval set.
Do not add `libero_goal_task0_eval` to training or normalization.

**Check return support before launching a GPU run:**

```bash
python -m toolkits.recap_libero_goal_preflight --stage returns
python - <<'PY'
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

root = Path("/data1/Muttaqien/rlinf_data/recap_libero_goal_task0")
train = root / "libero_goal_task0_train"
eval_set = root / "libero_goal_task0_eval"
stats = json.loads((train / "meta/stats.json").read_text())["return"]
lo, hi = float(stats["min"]), float(stats["max"])
assert np.isfinite([lo, hi]).all() and lo < hi <= 0, (lo, hi)
values = pq.read_table(
    eval_set / "meta/returns_libero_goal_task0_recap.parquet",
    columns=["return"],
)["return"].to_numpy()
outside = np.count_nonzero((values < lo) | (values > hi))
print(f"TRAIN return range: [{lo:.1f}, {hi:.1f}]")
print(f"EVAL return range: [{values.min():.1f}, {values.max():.1f}]")
print(f"EVAL frames outside TRAIN support: {outside}/{len(values)}")
if outside:
    raise SystemExit(
        "STOP: held-out targets exceed training normalization support; "
        "inspect return distribution before another GPU run."
    )
PY
```

**Run the diagnostic 1,000-step pilot first**, from the workstation's existing
Python environment with its known-good NVIDIA library/Ray configuration.
It does not modify either dataset or the previous checkpoint.

```bash
bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh \
  experiments/libero_goal_task0/recap_value_model_mixed_pilot
```

Inspect the held-out `eval/libero_goal_task0_eval/...` metrics at steps
250, 500, 750 and 1,000. Prioritize **eval MAE, eval Spearman, predicted
value std versus target std, and categorical accuracy**; training metrics
alone are not evidence of generalization. The first 500 steps include
learning-rate warmup, so a weak early result is not by itself decisive.
Also note that differently normalized categorical targets make raw losses
and MAE across the old and new critics imperfect direct comparisons.

**Only if pilot validation is promising**, start the separate full run:

```bash
bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh \
  experiments/libero_goal_task0/recap_value_model_mixed_full
```

The full experiment starts from scratch (it does **not** implicitly resume
the pilot). Locate its checkpoint under the
`recap_value_model_mixed_full-...` run directory, not the legacy
`recap_value_model_sft-...` directory:

```bash
find logs/value_sft -path '*recap_value_model_mixed_full-*/**/global_step_8000/actor/model_state_dict/full_weights.pt'
```

Before SIM-5, confirm that the checkpoint comes from the **mixed full
experiment** and passes held-out validation. The old 8,000-step
success-only checkpoint is for comparison only.

### Step 4 — SIM-5: Compute Goal-specific advantages

Set `GOAL_VALUE_CKPT` to the actual absolute `full_weights.pt` path:

```bash
GOAL_VALUE_CKPT=/absolute/path/to/GOAL_VALUE_MODEL/full_weights.pt

bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh \
  experiments/libero_goal_task0/recap_compute_advantages --nproc 4 \
  "advantage.value_checkpoint=$GOAL_VALUE_CKPT"

python -m toolkits.recap_libero_goal_preflight --stage advantages
```

The new label sidecar is
`libero_goal_task0_train/meta/advantages_libero_goal_task0_recap.parquet`.
Avoid using old `libero10_task0_recap` labels.

### Step 5 — SIM-6: Train the Goal-specific RECAP specialist

```bash
bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
  experiments/libero_goal_task0/cfg_rl_openpi
```

The Goal-specific policy initializes from the **existing original π₀.₅
LIBERO-SFT** model, not from the LIBERO-10 Task 0 RECAP specialist.

The planned budget is 3,000 optimizer steps. For a fair comparison against the
earlier shortened 3,000-step run, note that this Goal setup uses a 500-step
learning-rate warmup rather than the original 5,000-step policy warmup.
Document this distinction when interpreting performance.

### Step 6 — SIM-7: Compare the NEW Goal specialist against SFT

After training, choose the new Goal-specific `global_step_3000` checkpoint
**directory**:

```bash
GOAL_RECAP_CKPT=/absolute/path/to/GOAL_POLICY/checkpoints/global_step_3000
GOAL_EVAL=/data1/Muttaqien/rlinf_runs/cfg_rl/recap_goal_specialist_20episodes

python -m toolkits.recap_libero_compare_videos \
  --checkpoint "$GOAL_RECAP_CKPT" \
  --suite libero_goal --task-index 0 \
  --episodes 20 --seed 7 \
  --value-observations --resume \
  --output "$GOAL_EVAL"

python -m toolkits.recap_libero_value_curves \
  --output "$GOAL_EVAL" \
  --episodes 20 --seed 7 \
  --value-checkpoint "$GOAL_VALUE_CKPT"
```

**Do not overwrite**
`recap_sim7_value_comparison_20episodes_libero_goal`: that directory
contains cross-task transfer evaluation of the **LIBERO-10-trained** RECAP
policy and is useful as a separate scientific baseline.

To visualize the new specialist in the existing five-category browser dashboard,
restart the server using the new optional override (after completing evaluation):

```bash
python -m toolkits.recap_libero_web_dashboard \
  --output /data1/Muttaqien/rlinf_runs/cfg_rl/recap_sim7_value_comparison_20episodes \
  --goal-specialist-output "$GOAL_EVAL" \
  --episodes 20 --port 8765
```

With no `--goal-specialist-output`, the Goal tab still shows the earlier
**LIBERO-10-to-Goal transfer** evaluation. This preserves both datasets and
avoids giving them misleading identical labels.

The comparison to report is:

1. Original SFT baseline, Goal Task 0.
2. LIBERO-10-trained RECAP on Goal Task 0 (**transfer**, existing).
3. Goal-trained RECAP on Goal Task 0 (**specialist**, new).

All evaluations need the same LIBERO task, initial states, seeds, action
chunking and step budget to support a controlled comparison.

## Scientific limits and safety checks

- The SIM-4 critic is task-specific and is not the policy.
- The value-model preflight rejects corrupted/missing return sidecars.
- SIM-5 uses the newly trained Goal critic; no LIBERO-10 labels are copied.
- The learning pipeline does **not** run automatically as a background task.
- Training and evaluation are GPU jobs to be started on the workstation.
- Verify the newly collected sample quality, positive/negative outcome
  balance, and critic held-out metrics before trusting any improvement.
