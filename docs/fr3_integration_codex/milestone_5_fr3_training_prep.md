# Milestone 5 — FR3 π0.5 observation preparation

**Implemented:** A real-data, offline-only preparation step, building a full `(N,19)` candidate FR3 state array from the previously recorded FR3+Robotiq+two-camera raw episode. It reuses the existing strict raw episode validator and `candidate_19d_state`; it does **not** fabricate 7D action targets, success labels, clock calibration, norm stats, or an FR3-trained checkpoint.

Run on the existing 300-frame stationary recording, without ROS, GPU, model inference, or robot commands:

```bash
cd ~/Desktop/AIST-Tokyo/gamma/rlinf-project
git pull --ff-only origin branch-openpi-recap-franka
PYTHONPATH="$PWD" python -m toolkits.realworld_check.fr3_pi05_training_prep \
  --raw "$HOME/fr3_demo_20261107_084244" \
  --output "$HOME/fr3_pi05_observation_prep_20261107"
```

The NEW output folder contains `candidate_state_19d.npy`, `observation_monotonic_ns.npy`, and `report.json`. Original RGB frames remain under the raw input directory and are not duplicated. The output is **an observation preview, NOT a LeRobot training dataset**; never feed it to SFT or RECAP as labeled demonstrations.

**Why not train yet?** The original raw episode has `training_action=null` and zero operator events. Its robot states and images are valid but have no physically verified, controller-accepted 7D command/action labels, movement task outcomes, cross-host clock calibration, or supervised safety approval. Fitting normalization on stationary/unlabeled frames would be misleading. The StackCube Panda simulation policy used a normalized `pd_ee_delta_pose` command and a different 18D state; do not reuse its norm stats or actuation semantics for FR3.

**Next real requirement:** approved FR3 task demonstrations with actual accepted target/action provenance, verified tool/frame/gripper semantics, two-camera synchronization, and task outcomes. Then choose/freeze the physical 19D+7D dataset contract and derive the matching FR3 normalization before SFT. No robot motion or model training occurs here.
