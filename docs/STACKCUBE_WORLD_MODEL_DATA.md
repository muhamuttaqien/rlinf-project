# StackCube-v1 → shared Wan/OpenSora dataset: Phase 2

This is a **data preparation stage**, not Wan/OpenSora training. It preserves
existing RLinf training configurations and does not claim LIBERO checkpoints work
on ManiSkill.

## 1. Verify the existing Franka Panda simulator

```bash
python scripts/smoke_test_stackcube.py --robot panda --obs-mode rgb --image-size 256 --steps 10
```

## 2. Collect ManiSkill Panda StackCube demonstrations

Use ManiSkill's motion-planning example, rather than random zero-action rollouts.
Depending on the installed ManiSkill version, its CLI arguments and output
directory may vary. Check them before launching:

```bash
python -m mani_skill.examples.motionplanning.panda.run --help
python -m mani_skill.examples.motionplanning.panda.run -e StackCube-v1 --num-traj 10 --save-video
```

**Important:** Motion-planning trajectory recordings may contain only actions
and states. A successful motion-planning recording does *not* imply it contains
camera RGB. For state-only recordings, replay with ManiSkill's version-matched
trajectory replay facility using RGB observations and verify the resulting HDF5
contains `traj_N/obs/sensor_data/base_camera/rgb`. Do not treat a video MP4
as a substitute for action-aligned observations.

The exact capture/replay command depends on the ManiSkill version. Inspect
`python -m mani_skill.trajectory.replay_trajectory --help` and confirm its
camera, observation, and control settings before exporting.

## 3. Export action-aligned RGB trajectories

```bash
python scripts/export_stackcube_world_model.py \
  --input /path/to/rgb_recording.h5 \
  --output data/stackcube_world_model \
  --camera base_camera \
  --image-size 256
```

Each `traj_N.npz` contains `rgb` of shape `[T+1,256,256,3]` and
`actions` of shape `[T,7]`. `manifest.json` lists each sequence. The
exporter fails clearly on missing RGB, mismatched time axes, and incorrect
action dimensions, rather than silently mixing invalid data.

**Not yet implemented:** converting these sequences to Wan/OpenSora's
model-specific initialization / keyframe formats; preprocessing and training
the action-conditioned video model; training a StackCube reward classifier;
or running RLinf GRPO with the learned simulator. The existing LIBERO
checkpoints and action normalizers are **not** interchangeable with StackCube.

## Test (CPU; does not require ManiSkill)

```bash
python -m pytest tests/unit_tests/test_stackcube_world_model_export.py -q
```

Camera acquisition uses ManiSkill `sensor_configs={"width": 256, "height": 256}` in the smoke test. The exporter also resizes any recorded frames to 256x256 using bilinear filtering. This resizing alone does not make ManiSkill recordings compatible with Wan/OpenSora checkpoints.

## Verified local StackCube recording (2026-10-20)

The local Panda motion planner produced a 100-episode replay with 7-D
`pd_ee_delta_pose` actions and 256x256 RGB observations from `base_camera`
and `hand_camera`:

```bash
python scripts/export_stackcube_world_model.py \
  --input demos/StackCube-v1/motionplanning/stackcube_100.rgb.pd_ee_delta_pose.physx_cpu.h5 \
  --output data/stackcube_world_model \
  --camera base_camera \
  --image-size 256
```

Inspect `manifest.json` and verify every source episode's final `success`
flag before using the exported sequences for training. The exporter currently
validates dimensions but does not filter unsuccessful episodes or retain reward
labels. Its NPZ files are an intermediate format, not Wan/OpenSora training inputs.
