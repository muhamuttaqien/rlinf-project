# Milestone 6.2 — RECAP Critic Readiness and Training Strategy

**Status: PARTIAL.** The implementation and data path are now audited and the real StackCube pilot loads through RLinf's value dataset. The project has a supported route to train a critic from pretrained SigLIP2/Gemma3 backbones, but neither those backbones nor a trained value critic are available locally. Meaningful value predictions and advantages therefore remain unverified. No critic training, additional rollout collection, policy optimization, download, or source modification was performed.

## Scope and Starting State

The audit remained on `branch-openpi-recap-codex` at `b2bd05c3` (two commits ahead of its origin tracking branch). The worktree already contained the Milestone 6.1 source, test, and documentation changes, plus its untracked report; those changes were preserved. The Milestone 4 policy checkpoint, expert dataset, rollout data, and prior evaluation artifacts were not modified.

The visible Milestone 6.2 request ends mid-sentence at “Read source code rather…”. This report completes the specified algorithm, input, target, objective, data, resource, and readiness audit; additional requirements from the missing continuation could not be applied.

## Algorithm and Local Implementation

The RECAP method in the π*0.6 paper trains a language-conditioned distributional value function on empirical trajectory returns, then uses value estimates and rewards to label advantages and trains an advantage-conditioned policy. The critic predicts a distribution over 201 return bins; the scalar value is the expectation over those bins. The policy stage is advantage-conditioned supervised extraction with classifier-free guidance, rather than a PPO-style actor-critic policy-gradient update. The paper's full recipe uses demonstrations together with rollout and correction experience, iterating between data collection and offline optimization. See the [π*0.6 paper](https://arxiv.org/abs/2511.14759) and its [official paper PDF](https://www.pi.website/download/pistar06.pdf).

RLinf implements a four-stage pipeline: return sidecar generation, value-model supervised training, N-step advantage labeling, and CFG policy training. Its current StackCube critic path is image and prompt conditioned. Although the data adapter preserves the 18D Panda state and 7D action fields up to the current dataset/collator path, `ValueCriticModel` only uses image tensors, image masks, and prompt tokens. The paper describes observations that can include robot configuration, so image/text-only StackCube conditioning is a project-specific modeling choice and should be evaluated explicitly; adding state to the critic would change its input architecture and is not a mapping-only fix.

The local return processor uses synthetic task rewards: -1 at ordinary steps, 0 at a successful terminal, and the configured failure penalty at an unsuccessful terminal. It does not use the native ManiSkill dense rewards retained in the source dataset. The current advantage implementation combines a normalized N-step reward sum with `gamma^N * V(o[t+N]) - V(o[t])`; its episode-boundary behavior sets the next value to zero. The policy config uses advantage-conditioned CFG training. These implementation details are verified from `compute_returns.py`, `compute_advantages.py`, `cfg_rl_openpi.yaml`, the value model, and the RLinf RECAP guide; they should not be conflated with all possible choices in the paper.

The existing pilot's 150-step time-limit failures are assigned the configured failure terminal penalty by the return stage. The current pipeline treats an episode end as a zero-bootstrap boundary; it does not bootstrap a time-limit truncation. This is the local implementation's behavior. Changing it requires an explicit algorithm/data-label decision and consistent reprocessing, not a silent data adapter adjustment.

## Critic Architecture, Initialization, and Target

`ValueCriticModel` predicts categorical distributions over 201 bins in `[-1, 0]`; training uses a cross-entropy target with interpolation to adjacent bins. The local model config uses SigLIP2-so400m and Gemma3-270M pretrained backbones, plus RLinf's critic expert, projection, and value head. With no `model_path`, model construction loads these pretrained backbones and initializes the critic-specific components for training. A randomly initialized critic head is therefore a training initialization, not a usable pretrained value model. Advantage labeling requires a trained compatible critic checkpoint.

The configuration defaults to full VLM training, BF16, a micro-batch of 32, global batch 256, and up to 30,000 epochs unless bounded with `max_steps`; these are example defaults, not a defensible StackCube run budget. It also requests `balance_weights: true`, which balances configured dataset sources, not success labels within a source. The train worker derives a common return range from source statistics unless configured explicitly. Any mixed-source run must ensure that all return sidecars and normalizers use the same reward semantics and scale.

No compatible local SigLIP2/Gemma3 assets or trained RECAP critic checkpoint were found in the project outputs or inspected Hugging Face cache. The π₀.₅ SFT policy checkpoint is not a critic checkpoint. No weights were loaded and no value inference was run. The RLinf [RECAP implementation guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html) documents the required pretrained vision/language assets and value-training step.

## Real Dataset and Numeric Checks

The bounded CPU check instantiated `ValueDataset` on `outputs/stackcube_recap/milestone6_rollouts/returns_processed` with the StackCube adapter and decoded a real sample. It reported 1,500 rows and produced the expected mapped camera fields, prompt, and normalized target. Existing Milestone 6.1 checks also decoded both saved rollout camera videos and exercised image preprocessing; they did not perform a model forward pass.

The expert metadata contains 10 successful episodes with lengths `[107, 94, 101, 109, 109, 120, 119, 101, 110, 114]`, totaling 1,084 transitions. Recomputing the current return function on those lengths gives success return targets from -119 to 0. The 10 rollout failures each have 150 transitions and targets from -449 to -300 at `gamma=1`, `failure_reward=-300`. Under the current default normalization by the absolute global minimum (-449), successful expert targets lie in approximately `[-0.265, 0]`, while failed rollout targets lie in `[-1, -0.668]`. The success/failure return ranges are separated by this reward design; whether the visual critic can generalize from only these trajectories is unknown.

The existing 6.1 advantage arithmetic test uses supplied numeric values to validate formula and boundary handling. Those inputs are fixtures, not critic predictions. Accordingly, no meaningful model-derived advantages were calculated.

## Scientific Readiness Assessment

The current 10 failed policy trajectories alone do not provide observed successful autonomous outcomes. With the configured top-30% threshold, relative ranking can still designate a subset of failures as positive; “positive” then means top-ranked under the current critic/advantage distribution, not successful or task-positive. This makes value calibration and inclusion of genuine successful behavior particularly important.

The 10 expert trajectories are verified successes and can provide return supervision under RLinf's `type: sft` path, which labels expert demonstrations as successful. Combining expert and autonomous data is consistent with the broad RECAP data mixture described in the paper and RLinf guide, but the exact source weighting, split, and policy-extraction use are experimental choices. The current ten-plus-ten trajectory pool is too small for a persuasive generalization claim. A held-out split by complete episode is required to avoid temporal leakage; with only ten episodes per source, any validation metric will have high uncertainty. Report rank correlation and categorical calibration by source/outcome, not just aggregate loss.

The rollout dataset retains native dense rewards, but current value targets discard them and substitute sparse-like task labels. For StackCube, compare a preregistered task-level return design with native reward-to-go in a separate design decision before critic fitting. Do not replace the current labels in place. Validate success, failure, and time-limit semantics against source flags; distinguish `terminated` from `truncated`, and decide explicitly whether truncation bootstraps or receives a failure penalty.

RLinf's current critic does not use state or actions. This may be sufficient if images and task text make the value observable, but that assumption has not been shown for Panda manipulation. A state-conditioned critic would need model/API changes and a controlled comparison. Separately, the downstream CFG policy path must be checked for compatibility with the existing π₀.₅ SFT checkpoint before the project can claim end-to-end RECAP readiness; critic readiness alone does not establish advantage-conditioned policy training compatibility.

## Resource Estimate and Proposed Validation Sequence

The latest available workstation snapshot in this audit reported 112 GB free disk and 472 GiB host memory available. A fresh GPU snapshot was not taken because this milestone performs no GPU work. The configured backbone sources total several gigabytes before cache duplication and trained checkpoints; the official SigLIP2-so400m file listing is about 4.54 GB, with Gemma3-270M weights adding further storage. The available disk appears sufficient for one provenance-checked asset copy and outputs, but current space must be rechecked before downloading. Keep a disk margin because Hugging Face caches and saved checkpoints can duplicate weights.

The model config is BF16 but trains the backbone by default. Exact peak VRAM and checkpoint size cannot be established from config alone; activations, optimizer state, FSDP wrapping, and checkpoint serialization affect both. The four GPUs have ample nominal capacity, but a shared-workstation snapshot is stale quickly. When critic training is authorized, first load one model on one currently available GPU, measure memory, then run a short bounded optimization pilot with micro-batch 1–4 and a fixed step budget before setting a longer schedule. Do not use the stock 30,000-epoch / batch-32 defaults without a dataset-based bound.

Recommended next sequence, subject to approval:

1. Decide whether the critic stays image/text-only or receives the Panda state, and approve the reward/timeout semantics.
2. Obtain documented, compatible SigLIP2 and Gemma3 assets; record exact revisions and hashes. Keep the value head randomly initialized only as a supervised training start.
3. Build immutable return-processed expert and rollout copies with common semantics. Split by whole episodes and report source-stratified outcomes.
4. Run a bounded one-GPU critic training pilot with validation by trajectory, checking finite loss/gradients, categorical calibration, rank correlation, and overfit gaps. Save and reload the resulting critic before computing advantages.
5. Verify the OpenPI π₀.₅ CFG policy interface separately before planning any policy update.

No scaled collection or critic/actor training was started in this milestone.

## Validation and Repository Changes

- `PYTHONPATH=$PWD HF_DATASETS_CACHE=/tmp/rlinf-hf-datasets .venv/bin/pytest -q tests/unit_tests/test_stackcube_recap.py` — **6 passed**, 17 upstream deprecation warnings, 10.23 seconds. These are existing Milestone 6.1 tests; no new test was added in 6.2.
- Bounded CPU `ValueDataset` read against the real processed pilot — **passed**, 1,500 rows; sample fields and target loaded. This confirms preprocessing access only, not critic inference.
- Independent CPU return calculation from expert episode lengths — **passed**, 10 episodes/1,084 transitions and the ranges above.
- GPU use: none. Critic checkpoint load/inference: not run because compatible model weights are unavailable.
- Disk and host-memory measurements: 112 GB free and 472 GiB available at the captured snapshot; not a live reservation.

No Milestone 6.2 code or configuration changes were made. Existing Milestone 6.1 changes remain uncommitted and untouched. No commit or push was made. The branch remains `branch-openpi-recap-codex` at `b2bd05c3`.

## Remaining Blockers

1. No pretrained SigLIP2/Gemma3 source assets or trained StackCube value checkpoint are available locally.
2. Image/text-only critic sufficiency for this task is unvalidated; the value model currently ignores the 18D state.
3. The rollout pool has no native successes and only ten trajectories. Relative advantage quantiles can rank failures as positive.
4. Timeout/failure penalty and native dense reward semantics need an explicit experimental choice.
5. Critic generalization and held-out evaluation are weakly identifiable with the current number of episodes.
6. π₀.₅ CFG-policy compatibility remains a separate downstream verification requirement.

**Recommended next milestone:** approve critic experiment design and asset provenance, then perform a bounded value-model initialization/training pilot and checkpoint reload validation. Do not compute policy advantages or launch RECAP policy optimization until that critic and the downstream CFG interface are verified.
