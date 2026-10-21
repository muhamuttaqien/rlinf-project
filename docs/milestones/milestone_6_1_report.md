# Milestone 6.1 — StackCube RECAP Value and Advantage Integration

**Status: PARTIAL.** The StackCube dataset now has mappings through RLinf's existing value-data and advantage-inference input transforms, and its actual 1,500-row return sidecar passes structural and numerical checks. The categorical value critic cannot be loaded or used: this workspace has no compatible SigLIP2/Gemma3 backbones or trained critic checkpoint. The current critic also consumes images and task text only; it does not use the retained Panda state or action fields. Therefore no model-derived value estimates or meaningful dataset advantages were generated, and RECAP training readiness is not established.

## Scope and repository state

- Branch: `branch-openpi-recap-codex`; starting tracked worktree was clean. The branch was already two commits ahead of `origin/branch-openpi-recap-codex` from Milestone 6. No remote operation was performed.
- The 10-episode Milestone 6 LeRobot pilot and its return-processed copy were preserved. No model, dataset, normalization file, or system configuration was changed.
- The task was to integrate StackCube observations into the existing RLinf RECAP path and validate returns/advantages without starting critic or policy training.

## Algorithm and value-model audit

The existing implementation follows the four-stage RLinf RECAP workflow: return labeling, value-model supervised training, advantage labeling, then policy optimization. The return stage creates synthetic task-level labels (`-1` at nonterminal steps, `0` at successful terminal, configured failure reward at unsuccessful terminal); it does not use the pilot's native ManiSkill reward. The pilot's configured failure reward is `-300`, `gamma=1` in the Milestone 6 return run. The official [RLinf RECAP guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html) describes value SFT from pretrained SigLIP2 and Gemma3 backbones, a categorical value head, and advantage estimation from N-step rewards and adjacent value predictions.

The critic's existing `ValueCriticModel` input path processes camera images and prompt tokens. Its forward/predict path does not consume proprioceptive state or action. State/action fields remain useful and preserved in the transformed dataset sample and for policy training, but they do not affect this critic's value estimate. Extending the critic to condition on the 18D state would change model architecture and require critic training; that change was outside this integration and was not made.

Advantage labeling uses episode-aligned `V(o_t)` and `V(o_{t+N})` predictions and the configured N-step reward sum, with episode boundary handling that sets the next value to zero at an episode end. The existing pipeline's horizon handling treats a truncated episode as an episode end; it does not bootstrap through time-limit truncation. All ten pilot episodes are 150-step time-limit truncations and native failures, so the current task-level labeling creates failure terminal rewards for these data. This is the current implementation's semantics, not a general time-limit bootstrap rule.

The value training path requires pretrained visual/language backbones (SigLIP2-so400m and Gemma3-270M in the documented configuration); the critic-specific projection/expert/value head are trained separately. Advantage labeling additionally requires a compatible trained critic checkpoint. Searches of the documented local Hugging Face/OpenPI caches, project checkpoint directories, and `outputs/stackcube_recap` found the π₀.₅ policy checkpoints only; there was no SigLIP2/Gemma3 pair and no RECAP value/critic checkpoint. The π₀.₅ policy checkpoint is not a value-model substitute. No random critic was instantiated and no checkpoint was downloaded.

## StackCube adapter changes

Added StackCube mappings in `rlinf/models/embodiment/value_model/recap/checkpoint_utils.py` for:

- Value training samples from the LeRobot `task` field.
- Advantage inference samples, where `build_obs` supplies `prompt`.
- LeRobot source columns into value-inference image/state/action/prompt keys.

Registered `maniskill_stackcube` in the value dataset repack map, `build_input_transforms`, and advantage-labeling key map. The mapping routes `base_camera` to `base_0_rgb`, `hand_camera` to `left_wrist_0_rgb`, the unchanged 18D `observation.state` to `state`, 7D `action` to `actions`, and task text to prompt. Existing padding transforms extend state/actions to the model's 32D interface without changing their source dimensions or normalization. StackCube skips policy normalization in value inference, consistent with the value-training transform path; return-label normalization remains the existing RECAP behavior.

The standard value processor accepts the two mapped cameras and creates its established missing `right_wrist_0_rgb` placeholder with a false mask. That placeholder is not a third observed StackCube stream. The adapter retains state/action before the current value dataset/collator boundary, but the critic itself subsequently receives only images, image masks, prompt tokens, and value targets.

## Real-data processing and numeric validation

The CPU integration smoke used the actual files under `outputs/stackcube_recap/milestone6_rollouts/` and is summarized in `outputs/stackcube_recap/milestone6_1_recap/data_integration_smoke.json`.

- Return sidecar: 1,500 rows, 10 episodes, 150 contiguous frames per episode; all source episode rows align by `episode_index` and `frame_index`.
- Source native outcomes: 0 successes, 10 truncated at step 150, 0 simulator terminations. The return processor sidecar does not carry `is_success`; native flags are validated against the original pilot dataset.
- Each unsuccessful episode's generated RECAP rewards are `-1` for the first 149 steps and `-300` on the final step. With `gamma=1`, reward-to-go is `-449` at the first step and `-300` at the last step. These agree with the official processor's per-episode calculation.
- One actual parquet observation's state/action values passed through the value-training transform and advantage-inference mapping. The 18D state and 7D action were preserved in the first dimensions of padded 32D arrays.
- The actual episode-0, frame-0 base and hand camera images were decoded from the pilot MP4 videos (128×128 RGB) and processed under their mapped keys. The existing RECAP image processor produced finite `[1,3,224,224]` tensors for both cameras and the masked missing-camera placeholder. This smoke validated preprocessing only, not a critic model forward pass.

Focused tests compare success/failure return labels to independent numeric references, exercise N-step advantage arithmetic and episode-boundary `V_next=0`, and validate both StackCube transform paths. The advantage arithmetic fixture supplies known `value_current` numbers solely to check the existing computation; it is not a value-model output and is not evidence of meaningful advantages. No dataset advantages were written.

## Validation

- `PYTHONPATH=$PWD HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets .venv/bin/pytest -q tests/unit_tests/test_stackcube_recap.py` — **6 passed**, 17 upstream deprecation warnings.
- `.venv/bin/ruff check` on changed Python files — **passed** (Ruff emitted existing configuration warnings for preview-only DOC rules).
- `git diff --check` — **passed**.
- Real-data CPU preprocessing and return alignment smoke, including actual decoded camera frames — **passed**, summary saved at `outputs/stackcube_recap/milestone6_1_recap/data_integration_smoke.json`.
- GPU use: none. Critic/model loading was not attempted because required compatible weights were absent. No GPU memory or utilization measurement applies to this CPU-only validation. The integration smoke's aggregate wall time was not instrumented; the focused pytest suite completed in 10.25 seconds.

During smoke-harness development, two invalid attempts exposed harness assumptions (processor images must be tensors, then incorrect processor class import); another check assumed `is_success` existed in the return sidecar, while the source outcome flags are in the original dataset. These harnesses did not modify source artifacts and are not counted as successful checks. The final smoke explicitly reads source outcomes and return labels separately. A transient list-versus-array test input was corrected to match LeRobot's tensor/array transform contract; no production code change was needed for it.

## Readiness and remaining work

The existing rollout dataset is compatible with the return-labeling stage and the new StackCube mapping reaches the supported image/text value preprocessing interface. It is **not** sufficient to claim full value/advantage compatibility or RECAP training readiness because:

1. No local compatible SigLIP2/Gemma3 assets or trained value critic checkpoint were found.
2. A critic must be trained with the supported value-SFT stage before meaningful value predictions can label advantages.
3. The current critic does not use the preserved 18D proprioception; deciding whether image/text-only values are scientifically appropriate for this Panda setup, or extending the critic, needs an explicit design decision.
4. All ten rollout episodes failed. Native rewards are present in the source dataset, but the current RECAP return processor replaces them with task-level labels. The returns are therefore all failure-targeted; they provide no successful policy outcome contrast.
5. Ten failures are a limited value-training set. Mixing successful expert trajectories is supported by the documented RECAP workflow for value SFT, but the correct labels, split, and use of expert data must be verified for this StackCube setup. Balanced sampling is not assumed to be an algorithm requirement and was not implemented.

The next defensible step is an approved value-model preparation design: establish authorized/provenanced SigLIP2 and Gemma3 assets, decide image/text-only versus a separately designed proprioceptive critic, define how successful expert and failed rollout labels are combined and held out, then run a bounded value-SFT validation. Do not compute dataset advantages or start policy RECAP optimization until a compatible trained critic is verified.

## Changes and Git

Changed tracked files:

- `rlinf/data/datasets/recap/value_dataset.py`
- `rlinf/models/embodiment/value_model/recap/checkpoint_utils.py` (StackCube key maps and inference transform)
- `examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py`
- `tests/unit_tests/test_stackcube_recap.py`
- `docs/CODEX_EXPERIMENT_PROGRESS.md`
- `docs/milestones/milestone_6_1_report.md`

Diagnostic JSON remains under the Git-ignored `outputs/stackcube_recap/milestone6_1_recap/` directory. No commit or push has been made at report time. No training, further rollout collection, or policy update was started.
