# StackCube π₀.₅ Clean V1 — Implementation Audit (2026-10-23)

**Scope:** Data collection → persisted LeRobot data → real OpenPI SFT loader → transforms / loss → checkpoint → inference → ManiSkill EE controller. This is a source-code audit of `experiment/stackcube-pi05-restart-v1`, combined with reported workstation checks. It is not a claim that all simulator or upstream OpenPI internals were independently run.

**Definitions:** **Verified** = specific source contract corroborated by a reported workstation check or directly established by source. **Potential Issue** = real uncertainty, design risk, or insufficient direct evidence; *not* a proven implementation defect. **Confirmed Bug** = observed incorrect behavior supported by logs and source, with scope specified.

## Executive verdict

The Clean V1 baseline has measurably stronger engineering controls than the previous experiment. The previous train-only image crop is disabled; training and evaluation OpenPI input fields/geometry matched on a saved sample; 100 new successful demonstrations passed persistence audit; the normalized 7D → padded 32D → decoded 7D round-trip passed; and the new SFT completed 600 steps. **No new confirmed core data/action-alignment or transform bug has been identified from this audit.** A deterministic batch from the actual installed OpenPI SFT dataloader has now passed end-to-end parity checks, but this is not a proof about every training sample, multi-worker shuffling, or learned closed-loop policy performance.

Closed-loop results: **0/10 successes, 8/10 ever-grasped** at chunk=10; **0/3 successes, 2/3 ever-grasped** at chunk=1. An independent expert succeeds on evaluation seed 10000 in 63 steps.

## Component classifications

| Component | Classification | Grounded evidence | Limitation / action |
| --- | --- | --- | --- |
| Expert collector and pre-step observation/action alignment | **Verified** (source-level alignment and successful replay) | `collect_stackcube_ee_delta.collect_episode.apply` obtains `extract_ee_observation(env, obs)` **before** `env.step(action)`, then writes precisely that command and subsequent success. Only successful episodes are saved. Held-out expert seed 10000 succeeds. | This verifies capture logic, not a per-action numerical check of installed controller kinematics. |
| LeRobot persistence and success labeling | **Verified** (reported runtime) | User reported fresh 100-episode collection, `audit_stackcube_pi05_clean_data.py` passed, followed by fresh normalization stats. The audit checks 8D/7D columns, per-episode contiguous frame indices, only terminal success=True, and three independent base/wrist image pairs. | Three image pairs are a deliberately bounded inspection, not pixel-level inspection of every frame. |
| Episode-local action horizon | **Verified** (source) | `v3_stackcube_dataset.V3StackCubeDataset.__getitem__` constructs action indices `idx..idx+9`, clipped to that episode's last frame using `_last_index`. Frame and episode bounds are checked when loading. | Tail padding repeats the terminal action. This is explicit, not a confirmed off-by-one bug. |
| Actual OpenPI SFT loader end-to-end | **Verified** (deterministic real batch; reported runtime) | `official_sft_data_loader.py` uses the official OpenPI `create_data_loader` with a scoped v3 dataset factory override; it checks 10-step horizon, 32D model actions, and state-tokenization flags. Training ran 600 steps, so a real loader was exercised. | The actual SFT dataloader's unshuffled first batch was checked directly against persisted LeRobot row zero: state, images, masks, prompt tokens, normalized action target and decoded raw 7D actions matched. This tests one deterministic batch with microbatch 1 and num_workers 0, not the stochastic ordering of the training run. |
| Train/eval image geometry, prompt and state transform | **Verified** (tested sample, source) | Clean YAML sets `train_image_augmentation=False`; `get_model` passes it to `Pi0.sft_forward`; `compute_loss` uses it in `preprocess_observation`. `check_stackcube_pi05_clean_transforms.py` reported train/eval images/state/tokens identical. | Scope: deterministic geometry and tested saved sample, not a full-dataset parity proof. |
| Normalization, padding and output decoder | **Verified** (reported runtime) | Fresh norm stats loaded; round-trip 7D → normalized 32D → decoded 7D passed. `EnvIO.decode_actions` invokes `Unnormalize` and `ManiSkillOutputs(output_action_dim=7)`. | Round-trip establishes invertibility, not prediction quality. |
| SFT flow-matching objective | **Potential Issue** (design/training risk, not bug) | `Pi0.sft_forward(data, use_action_chunk_loss=False)` averages error over the full 32D model target; `FSDPVlaSftWorker.get_train_model_output` calls this without overriding the switch. Seven action channels are executable and the other 25 are padded. Diagnostic 7D losses are **not** the optimized objective. | Do not call this a coding defect without a controlled loss ablation. Training loss is a flow-matching loss, not direct action MAE. |
| Base weight initialization / checkpoint | **Verified** (code path); **Potential Issue** (provenance detail) | `get_model` accepts original `model.safetensors` or RLinf `full_weights.pt`; the evaluation successfully loaded the step-600 checkpoint. Clean YAML points to `PI05_MODEL_PATH`. | The exact **training startup log** naming the resolved original base weight file was not supplied to this audit. Verify that provenance if reproducing the experiment. |
| Inference transform and action ordering | **Verified** (tested paths) | `Pi0Eval.predict_action_batch` applies OpenPI env repack, sampling, and decoder; `evaluate_one` executes the predicted actions in order; unit tests verified 1 and 10-action chunk contracts. | Inference uses predicted observations after the first step; train-data parity does not guarantee closed-loop robustness. |
| Installed ManiSkill controller | **Verified** (reported preflight interface) | Live preflight passed: `pd_ee_delta_pose`, 7D action, normalized ±0.1 translation/rotation bounds, controller frame, 8D state, two cameras and seed repeatability. Expert action execution also succeeds. | It examines configured controller fields, not a separate numeric commanded-delta vs measured-delta replay. No evidence of a wrong sign convention in the live setup. |
| Gripper transition / lift imitation | **Potential Issue** (model or observation ambiguity) | With identical initial flow noise across adjacent expert frames and three seeds, the policy closes at training step 15 while expert remains open, then correctly closes at step 16. On held-out expert frame 31, expert z=+0.25 but policy z≈0. | Compare adjacent saved `state` and images at 14–17 to assess ambiguous observations. This is *not* evidence that dataset labels are shifted. Sampled frames alone do not establish global error rates. |
| Output action bounds | **Potential Issue** (low-level boundary check) | Evaluation reported actions slightly beyond normalized ±1 (e.g. gripper +1.008 and offline -1.02), and `evaluate_one` forwards decoded values to `env.step` without explicit clipping. | Whether this is consequential depends on the installed ManiSkill controller's handling of out-of-range values; no claim that it caused failure. |
| W&B step logging | **Confirmed Bug** (logging-only) | At shutdown W&B warns that step 599 metrics were logged after step 600 and ignored. This is an observed step-order defect in metrics emission, not proof of skipped optimizer steps. | Cosmetic/reporting issue; no reason to repeat the 600-step training. |
| Exact dataloader resume | **Potential Issue** (known limitation; not used in Clean V1) | `FSDPVlaSftWorker` persists dataloader cursor/RNG only for `StatefulDataLoader`. The wrapped official OpenPI loader does not obviously satisfy this condition. | Clean V1 was launched fresh. Do not advertise exact-resume reproducibility without a targeted resume check. |

## Gripper transition evidence

On training episode 0:
- At step 15, ground-truth gripper `+1`, predicted `-1.00`, `-0.84`, `-1.00` for noise seeds 1234–1236: consistently **premature closure**.
- At step 16, ground-truth `-1`, predicted `-1.01`, `-0.98`, `-1.01`: correctly closed.
- Step 14 predictions `+0.61`, `+0.91`, `+0.45`: sampling noise affects transition softness. The expert labels themselves change from `+1` to `-1` between steps 15 and 16.

This **does not validate a constant temporal shift**: the other frames align well. Consecutive observations may be nearly indistinguishable while the expert scripted controller switches phase; that hypothesis requires inspecting the saved adjacent states/images.

## Next decisive verification, without retraining

1. **Completed (PASSED):** One real batch returned by `build_openpi_sft_dataloader` matched persisted LeRobot row zero (camera pixels/masks, state, tokens, and all 10×32 model target values); decoding the actual loader target recovered 10×7 raw action values within tolerance.
2. **Only if needed:** Numerically compare training episode 0 observations at frames 14–17 (TCP pose/gripper width and base/wrist image differences). If observations are nearly identical, the gripper timing cannot reliably be supervised from a single visual/state frame without additional context.
3. **Optional, later:** A controlled 7D-only objective ablation or state/trajectory modeling change is a research experiment, **not** an implementation fix justified by current evidence.

**Stop condition cleared:** The actual loader-batch contract passed on the workstation. No new SFT or data collection is justified by a detected loader implementation defect. The next focused investigation is transition observability and motion-phase predictions; no large smoke-test campaign is warranted.

## Relevant source paths

- `toolkits/lerobot/collect_stackcube_ee_delta.py`
- `toolkits/lerobot/stackcube_ee_delta.py`
- `toolkits/lerobot/easy_stackcube_env.py`
- `scripts/audit_stackcube_pi05_clean_data.py`
- `rlinf/data/datasets/openpi/v3_stackcube_dataset.py`
- `rlinf/data/datasets/openpi/official_sft_data_loader.py`
- `rlinf/models/embodiment/openpi/dataconfig/maniskill_rlt_dataconfig.py`
- `rlinf/models/embodiment/openpi/transforms/pipeline.py`
- `rlinf/models/embodiment/openpi/pi0.py`
- `rlinf/models/embodiment/openpi/env_io.py`
- `rlinf/models/embodiment/openpi/tasks/eval.py`
- `scripts/evaluate_stackcube_pi05_ee_delta.py`
- `scripts/check_stackcube_pi05_clean_transforms.py`
- `scripts/diagnose_stackcube_pi05_clean_actions.py`

## Real SFT loader batch check (PASSED on workstation)

The new `scripts/check_stackcube_pi05_clean_sft_batch.py` calls
`build_openpi_sft_dataloader`, the same dispatch used by
`FSDPVlaSftWorker.build_dataloader`, with `eval_dataset=True` to disable
shuffling and make the *first* batch correspond to persisted LeRobot row zero.
For this one diagnostic, it uses `micro_batch_size=1` and `num_workers=0`;
these are sampling/execution settings, **not** an alternative implementation
of the SFT transforms. No model weights or GPU training are needed.

It checks the actual batch's state, all image pixels and camera masks,
prompt tokens, `[1,10,32]` normalized model action targets against
transforms applied independently to raw row zero; then decodes the **actual
loader target** back to the raw `[10,7]` action window. It cannot verify
that a shuffled multi-worker batch has the same ordering, because shuffled
ordering is intentionally excluded from this deterministic check.

Run from the project root on the original workstation (existing
`HF_LEROBOT_HOME`, `PI05_MODEL_PATH` environment):

~~~bash
PYTHONPATH="$PWD" python scripts/check_stackcube_pi05_clean_sft_batch.py \
  --repo-id local/stackcube_pi05_clean_v1 \
  --model-path "$PI05_MODEL_PATH" \
  --norm-stats /data1/Muttaqien/rlinf_data/pi05_stackcube_clean_v1_assets/local/stackcube_pi05_clean_v1/norm_stats.json
~~~

**Reported workstation output:** `OPENPI ACTUAL SFT BATCH PASSED dataset=local/stackcube_pi05_clean_v1 batch_actions=[1, 10, 32] decoded_actions=[10, 7]`. The first decoded 7D action matched the first raw 7D action within floating-point tolerance. The check compared the real loader's normalized model targets, camera pixels/masks, state, and prompt tokens against independently transformed row zero. Thus the **tested loader data path is Verified**, not a pending gap. This check is non-mutating and does not involve training, simulated rollouts, or new demonstrations.
