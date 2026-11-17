#!/usr/bin/env bash
# Compare LIBERO-Goal Task 0 RECAP policy checkpoints on 20 paired seed-7 episodes.
# Uses physical GPUs 1,2,3 by default; never launches work on GPU 0.
# Step-3000 results are reused, not recomputed.
#
# Run inside tmux from the repository root:
#   bash toolkits/run_recap_goal_checkpoint_comparison.sh
#
# Optional overrides:
#   GOAL_CFG_STEP3000=/absolute/path/to/global_step_3000
#   GPU1000=1 GPU1500=2 GPU2000=3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="/data1/Muttaqien/rlinf_runs/cfg_rl"
EXISTING="$DATA_ROOT/recap_goal_specialist_20episodes"
OUT_ROOT="$DATA_ROOT/recap_goal_checkpoint_comparison_seed7_20episodes"
mkdir -p "$OUT_ROOT"
[[ -s "$EXISTING/paired_results.csv" ]] || {
  echo "Missing completed step-3000 paired_results.csv: $EXISTING" >&2
  exit 1
}

if [[ -z "${GOAL_CFG_STEP3000:-}" ]]; then
  mapfile -t candidates < <(find "$DATA_ROOT" -type d \
    -path '*/recap_libero_goal_task0_cfg/checkpoints/global_step_3000' | sort)
  if (( ${#candidates[@]} != 1 )); then
    printf 'Found %d possible step-3000 checkpoints. Set GOAL_CFG_STEP3000 explicitly.\n' "${#candidates[@]}" >&2
    printf '%s\n' "${candidates[@]}" >&2
    exit 1
  fi
  GOAL_CFG_STEP3000="${candidates[0]}"
fi
CKPT_PARENT="$(dirname "$GOAL_CFG_STEP3000")"

for step in 1000 1500 2000 3000; do
  ckpt="$CKPT_PARENT/global_step_${step}/actor/model_state_dict/full_weights.pt"
  [[ -s "$ckpt" ]] || {
    echo "Missing or empty policy weights: $ckpt" >&2
    exit 1
  }
done

GPU1000="${GPU1000:-1}"
GPU1500="${GPU1500:-2}"
GPU2000="${GPU2000:-3}"
if [[ "$GPU1000" == "$GPU1500" || "$GPU1000" == "$GPU2000" || "$GPU1500" == "$GPU2000" ]]; then
  echo 'GPU assignments must be distinct.' >&2
  exit 1
fi
if [[ "$GPU1000" == 0 || "$GPU1500" == 0 || "$GPU2000" == 0 ]]; then
  echo 'GPU 0 is deliberately excluded due to other processes.' >&2
  exit 1
fi

export LD_LIBRARY_PATH="$HOME/.local/lib/nvml-fix:${LD_LIBRARY_PATH:-}"
echo "Policy checkpoints: $CKPT_PARENT"
echo "Existing step-3000 evaluation: $EXISTING"
echo "New outputs: $OUT_ROOT"
echo "GPU assignments: step1000=$GPU1000, step1500=$GPU1500, step2000=$GPU2000"
echo 'Note: GPU 1 previously showed an Xid watchdog timeout; monitor its run.'

declare -a pids=()
declare -a steps=(1000 1500 2000)
declare -a gpus=("$GPU1000" "$GPU1500" "$GPU2000")
for i in "${!steps[@]}"; do
  step="${steps[i]}"
  gpu="${gpus[i]}"
  output="$OUT_ROOT/step_${step}"
  logfile="$OUT_ROOT/step_${step}.log"
  echo "Starting step ${step} on physical GPU ${gpu}; log: $logfile"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -m toolkits.recap_libero_compare_videos \
      --checkpoint "$CKPT_PARENT/global_step_${step}" \
      --suite libero_goal --task-index 0 \
      --episodes 20 --seed 7 --value-observations --resume \
      --output "$output"
  ) > "$logfile" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[i]}"; then
    echo "Finished step ${steps[i]}: $OUT_ROOT/step_${steps[i]}/paired_results.csv"
  else
    echo "FAILED step ${steps[i]}; see $OUT_ROOT/step_${steps[i]}.log" >&2
    status=1
  fi
done
if (( status )); then
  echo 'At least one evaluation failed. Rerun this script to resume completed episodes.' >&2
  exit 1
fi

python - "$EXISTING" "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path
import pandas as pd

existing = Path(sys.argv[1])
out_root = Path(sys.argv[2])

def read_rows(directory):
    df = pd.read_csv(directory / "paired_results.csv").sort_values("episode")
    assert len(df) == 20 and df["episode"].tolist() == list(range(1, 21)), directory
    assert df["seed"].tolist() == list(range(7, 27)), directory
    return df

reference = read_rows(existing)
summary = []
for step, directory in [(1000, out_root / "step_1000"),
                        (1500, out_root / "step_1500"),
                        (2000, out_root / "step_2000"),
                        (3000, existing)]:
    df = read_rows(directory)
    if not df["baseline_success"].equals(reference["baseline_success"]):
        raise SystemExit(
            f"Baseline episode outcomes differ from step-3000 for step {step}; "
            "investigate reproducibility before comparing checkpoint success rates."
        )
    baseline = int(df["baseline_success"].sum())
    recap = int(df["recap_success"].sum())
    recap_only = int(((df.baseline_success == 0) & (df.recap_success == 1)).sum())
    baseline_only = int(((df.baseline_success == 1) & (df.recap_success == 0)).sum())
    summary.append({
        "step": step, "episodes": 20,
        "baseline_successes": baseline,
        "recap_successes": recap,
        "baseline_rate_pct": 5 * baseline,
        "recap_rate_pct": 5 * recap,
        "difference_pp": 5 * (recap - baseline),
        "recap_only_successes": recap_only,
        "sft_only_successes": baseline_only,
    })

result = pd.DataFrame(summary)
result.to_csv(out_root / "checkpoint_summary.csv", index=False)
print("\nLIBERO-Goal Task 0: 20 paired episodes, seed 7")
print(result.to_string(index=False))
print(f"\nSaved {out_root / 'checkpoint_summary.csv'}")
PY
