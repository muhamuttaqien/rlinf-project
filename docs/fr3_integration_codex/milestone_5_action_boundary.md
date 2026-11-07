# Milestone 5 — Non-actuating policy output boundary

**Purpose:** Connect the actual offline inference output to an auditable boundary without implying the StackCube Panda simulation policy can actuate FR3.

The candidate 7D FR3 semantics are TCP translation delta (m), rotation-vector delta (rad), and Robotiq target knuckle position (rad), but **these have not been verified to match the StackCube checkpoint's output semantics**. The actual NUC joint controller consumes normalized SpaceMouse motion intent; it has no proven command API for those metric 7D targets. There is **no legitimate automatic conversion**.

The module `toolkits/realworld_check/fr3_pi05_action_boundary.py` only inspects a finite `(N, 7)` model output and records its provenance and blockers. It exports **no ROS publisher**, no command generator, no control-enable switch. `to_fr3_controller_command` always refuses with `PermissionError`.

Previously reported workstation checks:

- Official PaliGemma tokenizer: **passed**, prompt tokens=6.
- Real recorded dual-camera observation + StackCube π0.5 checkpoint: **passed**, output `[1,10,32]`; first seven channels `[10,7]` finite.
- StackCube `norm_stats.json`: state 18, actions 7; cannot normalize the FR3 candidate 19D state. **Not applied**.
- Physical actuation: **not tested or approved**, commands sent=0.

## One quick offline test on the workstation

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" \
  python -m pytest -q tests/unit_tests/test_fr3_pi05_action_boundary.py
```

Expected: `5 passed`. This test requires only NumPy and pytest, not ROS, GPUs or robot motion. The checkpoint smoke is already proven; rerunning its heavyweight forward is **not required** just to validate the boundary. If the user needs a combined numerical report, rerun the existing checkpoint smoke and inspect the new `output_action_quarantine` field.

**Remaining Milestone 5:** determine task-appropriate FR3 policy/normalization, verify the action semantics and command target API, implement safety-reviewed explicitly authorized controller-side interface and default-disabled handoff. Physical validation belongs to Milestone 6.
