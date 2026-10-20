#!/usr/bin/env bash
set -euo pipefail

# ManiSkill motion-planning demonstrations for subsequent RECAP dataset preparation.
# Run from repository root with the embodied virtual environment activated.
COUNT="${1:-100}"
# Pass --replay-only to reuse a previously collected dataset without recollecting.
REPLAY_ONLY=false
if [[ "$COUNT" == "--replay-only" ]]; then
  REPLAY_ONLY=true
  COUNT=10
fi
BASE="${2:-outputs/stackcube_recap}"
if ! [[ "$COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Expected positive integer trajectory count, got: $COUNT" >&2
  exit 2
fi
mkdir -p "$BASE/raw" "$BASE/rgb"
if [[ "$REPLAY_ONLY" == false ]]; then
  echo "Collecting $COUNT successful StackCube expert trajectories..."
python -m mani_skill.examples.motionplanning.panda.run \
  -e StackCube-v1 \
  -n "$COUNT" \
  --only-count-success \
  -o none \
  --record-dir "$BASE/raw" \
  --traj-name stackcube_expert
fi

RAW="$BASE/raw/StackCube-v1/motionplanning/stackcube_expert.h5"
if [[ ! -f "$RAW" ]]; then
  echo "Trajectory not found at $RAW. Search the recording directory:" >&2
  find "$BASE/raw" -type f -name '*.h5' >&2
  exit 1
fi

echo "Collected expert trajectory file: $RAW"
echo "Replay in RGB with end-effector delta-pose control; only successful replays should be used for SFT."
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "$RAW" \
  --use-first-env-state \
  -c pd_ee_delta_pose \
  -o rgb \
  --save-traj

echo "Replay complete. Inspect the generated .h5 and .json metadata; replay may filter failed trajectories."
echo "These are ManiSkill trajectories, NOT LeRobot data yet."
