#!/usr/bin/env bash
# LIBERO-Goal Task 0: evaluate ONLY policy step 2000, split across GPUs 1,2,3.
# 20 paired seeds (7..26) are the same as the completed step-3000 experiment.
# Reuses the existing validated SFT baseline recordings; never reruns baseline.
# Each GPU gets a disjoint episode shard (7 / 7 / 6) and its own output directory.
#
# Run from the repository root inside tmux:
#   bash toolkits/run_recap_goal_checkpoint_comparison.sh
#
# Optional: GOAL_CFG_STEP2000=/absolute/path/to/global_step_2000
#           GPU1=1 GPU2=2 GPU3=3
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT=/data1/Muttaqien/rlinf_runs/cfg_rl
EXISTING="$DATA_ROOT/recap_goal_specialist_20episodes"
OUT_ROOT="$DATA_ROOT/recap_goal_step2000_parallel_20episodes"
mkdir -p "$OUT_ROOT"

[[ -s "$EXISTING/paired_results.csv" && -s "$EXISTING/run_config.json" ]] || {
  echo "Missing completed step-3000 evaluation and manifest under $EXISTING" >&2
  exit 1
}

# Derive the correct training run from the step-3000 evaluation manifest.
readarray -t manifest_fields < <(python - "$EXISTING/run_config.json" <<'PY'
import json
import sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
if (m.get("suite"), m.get("task_index"), m.get("episodes"), m.get("seed")) != ("libero_goal", 0, 20, 7):
    raise SystemExit("Step-3000 manifest is not LIBERO-Goal Task 0, 20 episodes, seed 7")
if not m.get("value_observations"):
    raise SystemExit("Step-3000 baseline lacks wrist/state recordings")
ckpt = Path(m["checkpoint"]).expanduser().resolve()
if ckpt.name != "global_step_3000":
    raise SystemExit("Expected step-3000 checkpoint in existing manifest")
print(str(ckpt.parent / "global_step_2000"))
print(str(Path(m["model_path"]).expanduser().resolve()))
PY
)
CKPT="${GOAL_CFG_STEP2000:-${manifest_fields[0]}}"
SFT_MODEL="${manifest_fields[1]}"
[[ -s "$CKPT/actor/model_state_dict/full_weights.pt" ]] || {
  echo "Missing step-2000 model weights: $CKPT" >&2
  exit 1
}
[[ -d "$SFT_MODEL" ]] || { echo "Missing SFT model: $SFT_MODEL" >&2; exit 1; }

GPUS=("${GPU1:-1}" "${GPU2:-2}" "${GPU3:-3}")
if [[ "${GPUS[0]}" == "${GPUS[1]}" ||
      "${GPUS[0]}" == "${GPUS[2]}" ||
      "${GPUS[1]}" == "${GPUS[2]}" ]]; then
  echo "GPU assignments must be distinct." >&2
  exit 1
fi
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[1-9][0-9]*$ ]] || {
    echo "GPU ID must be a positive integer; GPU 0 is excluded: $gpu" >&2
    exit 1
  }
done

export LD_LIBRARY_PATH="$HOME/.local/lib/nvml-fix:${LD_LIBRARY_PATH:-}"
echo "Evaluating ONLY LIBERO-Goal Task 0 policy step 2000: $CKPT"
echo "Using prior step-3000 SFT baseline: $EXISTING/baseline"
echo "Three physical GPU workers: ${GPUS[*]} (episodes 1-7, 8-14, 15-20)"
echo "Results: $OUT_ROOT"
echo "GPU 1 previously had an Xid watchdog fault. Inspect its log if it fails."

# The SIM-7 evaluator accepts 1-based --episode-indices, while --trials
# must stay 20 to preserve episode-specific LIBERO init states and seeds.
EP_GROUPS=("1,2,3,4,5,6,7" "8,9,10,11,12,13,14" "15,16,17,18,19,20")
declare -a pids=()
declare -a launched=()

for i in 0 1 2; do
  gpu="${GPUS[$i]}"
  shard="$OUT_ROOT/shards/shard_$i"
  mkdir -p "$shard"

  # Only rerun missing or corrupted recordings; quarantine partial artifacts.
  missing=$(python - "$shard" "${EP_GROUPS[$i]}" "$OUT_ROOT" "$i" <<'PY'
import shutil
import sys
from datetime import datetime
from pathlib import Path
from toolkits.recap_libero_compare_videos import _matching_files, _valid_existing_episode

shard = Path(sys.argv[1])
episodes = [int(e) for e in sys.argv[2].split(",")]
out_root = Path(sys.argv[3])
idx = sys.argv[4]
missing = []
for episode in episodes:
    if _valid_existing_episode(shard, episode, 7 + episode - 1, True, 0):
        continue
    partial = _matching_files(shard, episode, 0)
    if partial:
        backup = out_root / "interrupted_backup" / datetime.now().strftime("%Y%m%d_%H%M%S_%f") / f"shard_{idx}"
        backup.mkdir(parents=True, exist_ok=True)
        for item in partial:
            shutil.move(str(item), str(backup / item.name))
    missing.append(episode)
print(",".join(map(str, missing)))
PY
  )

  if [[ -z "$missing" ]]; then
    echo "GPU $gpu / shard $i: all assigned episodes already complete"
    continue
  fi

  echo "GPU $gpu / shard $i: evaluating episodes $missing"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -m toolkits.recap_libero_sim7_eval \
      --checkpoint "$CKPT" \
      --model-path "$SFT_MODEL" \
      --suite libero_goal --task-index 0 \
      --trials 20 --video-count 20 --seed 7 \
      --episode-indices "$missing" --value-observations \
      --output "$shard"
  ) > "$OUT_ROOT/shard_$i.log" 2>&1 &
  pids+=("$!")
  launched+=("$i")
done

status=0
for i in "${!pids[@]}"; do
  shard_id="${launched[$i]}"
  if wait "${pids[$i]}"; then
    echo "Finished shard $shard_id"
  else
    echo "FAILED shard $shard_id; see $OUT_ROOT/shard_$shard_id.log" >&2
    status=1
  fi
done
if (( status )); then
  echo "Some GPU evaluations failed. Rerun this script to resume only missing episodes." >&2
  exit 1
fi

# Reuse baseline MP4/wrist/state files via hard links (same /data1 filesystem).
# Never change the existing baseline or step-3000 comparison.
# Validate every shard before merging, and write a provenance manifest.
python - "$EXISTING" "$OUT_ROOT" "$CKPT" "$SFT_MODEL" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

from toolkits.recap_libero_compare_videos import _matching_files, _valid_existing_episode

existing, output, checkpoint, model_path = map(Path, sys.argv[1:])
checkpoint = checkpoint.resolve()
model_path = model_path.resolve()
old_manifest = json.loads((existing / "run_config.json").read_text())
expected_manifest = {
    "seed": 7, "episodes": 20, "model_path": str(model_path),
    "checkpoint": str(checkpoint), "value_observations": True,
    "suite": "libero_goal", "task_index": 0,
}
if (output / "run_config.json").exists():
    original = json.loads((output / "run_config.json").read_text())
    if any(original.get(k) != v for k, v in expected_manifest.items()):
        raise SystemExit("Output manifest mismatch: refusing to mix policy checkpoints.")
if Path(old_manifest["model_path"]).resolve() != model_path:
    raise SystemExit("Baseline model path mismatch.")
for episode in range(1, 21):
    shard_id = 0 if episode <= 7 else 1 if episode <= 14 else 2
    shard = output / "shards" / f"shard_{shard_id}"
    for folder, src in (("baseline", existing / "baseline"), ("recap", shard)):
        if not _valid_existing_episode(src, episode, 7 + episode - 1, True, 0):
            raise SystemExit(f"Incomplete {folder} recording for episode {episode}: {src}")

# All source episodes are complete, now stage links. No data is deleted.
for episode in range(1, 21):
    shard_id = 0 if episode <= 7 else 1 if episode <= 14 else 2
    shard = output / "shards" / f"shard_{shard_id}"
    for policy, src in (("baseline", existing / "baseline"), ("recap", shard)):
        dest = output / policy
        dest.mkdir(parents=True, exist_ok=True)
        for file in _matching_files(src, episode, 0):
            target = dest / file.name
            if target.exists():
                if not os.path.samefile(file, target):
                    raise SystemExit(f"Destination already exists with different data: {target}")
                continue
            try:
                os.link(file, target)
            except OSError:
                shutil.copy2(file, target)

manifest_path = output / "run_config.json"
if not manifest_path.exists():
    temp = manifest_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(expected_manifest, indent=2))
    temp.replace(manifest_path)
print("Validated and merged all 20 paired episodes into", output)
PY

# The original gallery composer checks every episode and generates a standalone
# paired_results.csv, comparison_all_episodes.mp4 and index.html.
python -m toolkits.recap_libero_compare_videos \
  --checkpoint "$CKPT" --model-path "$SFT_MODEL" \
  --suite libero_goal --task-index 0 \
  --episodes 20 --seed 7 --value-observations --resume \
  --output "$OUT_ROOT"

# Compare step 2000 against the already completed step 3000.
python - "$EXISTING" "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path
import pandas as pd

prior, current = map(Path, sys.argv[1:])
a = pd.read_csv(prior / "paired_results.csv").sort_values("episode").reset_index(drop=True)
b = pd.read_csv(current / "paired_results.csv").sort_values("episode").reset_index(drop=True)
assert len(a) == len(b) == 20
assert a["episode"].tolist() == b["episode"].tolist() == list(range(1, 21))
assert a["seed"].tolist() == b["seed"].tolist() == list(range(7, 27))
if not a["baseline_success"].equals(b["baseline_success"]):
    raise SystemExit("Baseline outcomes differ from existing experiment; investigate before reporting.")
rows = []
for step, df in ((2000, b), (3000, a)):
    base = int(df["baseline_success"].sum())
    recap = int(df["recap_success"].sum())
    rows.append({
        "checkpoint_step": step, "episodes": 20,
        "sft_successes": base, "recap_successes": recap,
        "sft_rate_pct": 100 * base / 20,
        "recap_rate_pct": 100 * recap / 20,
        "difference_pp": 100 * (recap - base) / 20,
        "recap_only_successes": int(((df.baseline_success == 0) & (df.recap_success == 1)).sum()),
        "sft_only_successes": int(((df.baseline_success == 1) & (df.recap_success == 0)).sum()),
    })
result = pd.DataFrame(rows)
result.to_csv(current / "checkpoint_summary.csv", index=False)
print("\nLIBERO-Goal Task 0: same 20 paired seeds (7..26)")
print(result.to_string(index=False))
print("Saved:", current / "checkpoint_summary.csv")
PY
