# Milestone 4C — raw data preflight and LeRobot export gate

## Scope and current status

**4C.1 COMPLETE: CPU-only tests and real 100-frame workstation preflight passed on November 7, 2026.**
This stage inspects existing 4A/4B raw FR3+Robotiq+two-camera captures;
it does not communicate with ROS hardware, act on the robot, or claim
a LeRobot training dataset exists.

Use `toolkits/realworld_check/fr3_lerobot_preflight.py` to:

- Verify the raw capture's manifest, counts, timestamps, contiguous
  frame indexes, original lossless images, their declared camera
  serials/resolutions, and measured-state shapes/finiteness.
- Check both camera images as `uint8` HWC `bgr8` and explicitly convert
  a sample to `rgb8` with `image`=base and `extra_view_image`=wrist.
- Preserve honest metadata: `is_lerobot_dataset=false`,
  `sft_export_ready=false`, `recap_export_ready=false`.
- Fail closed on missing frame, invalid source, corrupted raw values,
  missing view, incomplete capture, dropped events, or fabricated
  training-action labels.
- Optionally save `report.json` plus a first-frame
  `observation_preview_rgb.npz` to a **new** directory. This preview
  contains only the two image arrays, **no state or action labels**.

## Workstation validation

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" \
  python -m pytest -q \
    tests/unit_tests/test_fr3_lerobot_preflight.py \
    tests/unit_tests/test_fr3_raw_capture.py \
    tests/unit_tests/test_franka_pi05_contract.py

python -m toolkits.realworld_check.fr3_lerobot_preflight \
  --raw "$HOME/fr3_raw_buttons_20261106_183614" \
  --output "$HOME/fr3_4c_preflight_$(date +%Y%m%d_%H%M%S)"
```

The preflight validates the user's actual raw 100-frame pilot without
requiring RealSense access, ROS controllers, or operator motion. It may
take longer on full-resolution images than the small CPU-only unit
tests. The output preview is **not** a transform-stage sample:
`franka_pi05_contract.validate_realworld_pi05_sample` also requires
a justified 19D state and (N,7) action array.

## Verified real data preflight — November 7, 2026

The operator ran focused tests on the workstation in the
`aist-gamma` Python 3.12.3 environment:

```text
19 passed in 0.25s
```

- `test_fr3_lerobot_preflight.py`: 7 passed
- `test_fr3_raw_capture.py`: 5 passed
- `test_franka_pi05_contract.py`: 7 passed

Real source directory:
`/home/muttaqien/fr3_raw_buttons_20261106_183614`

The 4C preflight returned `raw_capture_valid=true` with **100**
validated state records, **1008** ROS events (1002 `operator_twist`,
6 `operator_gripper_request`), unique camera serials
(`337122075473` base, `218622277827` wrist), base
640×480 and wrist 848×480 input profiles and
`rgb8` preview conversion from original `bgr8`.

**Important:** The report explicitly returned
`is_lerobot_dataset=false`, `sft_export_ready=false`,
`recap_export_ready=false`, and
`training_action_labels_available=false`. This is a passing raw
data integrity preflight only; it does not establish a dataset for
training. The user's local preview directory was created by the
preflight, but the images have not been visually inspected by the
assistant.

**Next gate: 4C.2 — explicitly specify and qualify the 19D state
and 7D action semantics, including timing; no automatic projection
of the existing 20D Franka state or normalized SpaceMouse inputs.**

## Required next work — do not silently guess

1. **Episode semantics:** record genuine human-controlled task episodes,
   episode boundaries and human-assessed terminal/success/failure labels
   after local physical safety clearance. The November 6 captures are
   unlabeled stationary pilots.
2. **19D state contract:** choose and document the 19 numeric features,
   quaternion convention, gripper feedback/width calibration and units.
   RLinf's existing nested Franka state has 20 scalars; the raw
   recorder captures separate measured joint + TCP + wrench + knuckle
   angle fields. Neither is automatically a 19D policy state.
3. **7D action contract:** define Cartesian translation/rotation frame,
   units and interval plus gripper target; align to physically measured
   / executed control signals. SpaceMouse normalized `Twist` is **not**
   an executed physical action or a training target. The current
   controller uses normalized 0.007 m and 0.03 rad pose offsets in its
   IK solver, not a direct SI velocity contract.
4. **Cross-host timing:** document clock/exposure synchronization,
   action timestamp alignment and dropped-frame handling.
5. **Actual LeRobot export:** after the gates above are fulfilled,
   create episode/frame/action columns and matching metadata, save
   two camera streams with decoded RGB semantics, and validate
   real samples using the existing OpenPI contract.
6. **RECAP integration:** the current value dataset routes
   `franka` and `franka_co_train` through `FrankaEEInputs`, not
   `RealworldInputs`. Add a justified two-camera path and prove SFT,
   value/returns, advantages and CFG loader compatibility before
   declaring 4C complete.

No training, LeRobot dataset, or rollout labels have been generated by
this offline preparation step.
