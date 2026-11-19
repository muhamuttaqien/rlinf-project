# FR3 Peg Insertion: 10 Hz Dataset Preparation

Use this report to select the frequency-consistent dataset and review its proposed OpenPI π₀.₅ SFT configuration. The 100 existing demonstrations support an offline 10 Hz conversion. Training and physical evaluation remain separate steps.

## Recorded Data And Timing

The source is `/data1/Muttaqien/fr3_peg_100`, episodes `episode_0001` through `episode_0100`, collected on 2026-11-09. The audit reads every state row and command event, checks the original operator outcome reviews, and compares the existing converted labels with their provenance.

| Recorded stream | Samples | Median interval | Maximum interval |
| --- | ---: | ---: | ---: |
| SpaceMouse normalized Twist requests | 212,623 | 9.999813 ms | 13.844603 ms |
| Saved joint state | 21,234 | 100.030117 ms | 102.280906 ms |
| Saved TCP pose | 21,234 | 100.023062 ms | 102.933725 ms |
| Saved external wrench | 21,234 | 100.034575 ms | 103.327353 ms |
| Saved physical gripper state | 21,234 | 100.007965 ms | 103.347516 ms |
| Saved base RGB source | 21,234 | 101.179728 ms | 104.181191 ms |
| Saved wrist RGB source | 21,234 | 101.344827 ms | 104.165103 ms |
| Completed observation records | 21,234 | 100.126373 ms | 107.106894 ms |

Twist episode rates range from 99.9921 to 100.0086 Hz. Saved state averages approximately 9.988 Hz. All audited streams have strictly increasing timestamps. The 200 gripper requests are discrete button events, two per episode; an event rate is not a sampling frequency.

`fr3_data_collect.py` assembles observations at its 10 Hz default and configures cameras at 30 Hz: base serial `337122075473` at 640×480, wrist serial `218622277827` at 848×480. `fr3_raw_capture.py` saves selected BGR images and state snapshots, then stamps the completed record. Intervening camera frames and raw 100 Hz robot states are not present in these recordings. The NUC archive referenced by the reviews, `fr3_telemetry_isolated_4c2t7/recordings/stationary_telemetry.jsonl`, has a 2026-11-07 file modification time and steady-clock stamps without a usable per-episode clock link. It cannot establish synchronized high-rate state for the November 9 demonstrations.

All alignment measurements use workstation monotonic receipt times:

| Synchronization measurement | Median | 99th percentile | Maximum |
| --- | ---: | ---: | ---: |
| Base/wrist receipt skew | 9.386874 ms | 30.247218 ms | 33.973369 ms |
| Four state-topic receipt skew | 0.916603 ms | 2.047988 ms | 3.302566 ms |
| All six modality receipt skew | 23.780458 ms | 34.037181 ms | 34.710843 ms |
| Record completion after latest receipt | 10.035497 ms | 14.497150 ms | 17.241544 ms |

Camera receipts are host read-completion times, not calibrated exposure times. ROS-header-to-host-UNIX differences combine clock offset and transport delay; they do not establish one-way latency. Physical controller acceptance and tool-tip calibration cannot be inferred from these files.

## Action Meaning And Resampling

The selected labels come from `/franka_controller/target_cartesian_velocity_percent`: six already axis-mapped, normalized held SpaceMouse requests in linear `x,y,z`, angular `x,y,z` order. The recorded units explicitly say `normalized_controller_input_not_m_per_s` and `normalized_controller_input_not_rad_per_s`. Preserve their signs and scale 1.0. The controller applies its own mapping, limits, and acceptance logic downstream.

For each exact 100 ms interval, export the time-weighted mean of the held request:

`action[k] = integral(request(t), t[k], t[k] + 0.1 s) / 0.1 s`.

This preserves normalized request-seconds over the retained trajectory. It does not assert equality of physical robot motion after averaging nonlinear controller inputs. The measured-TCP delta proxy from `fr3_observed_action_proxy.py` is a different label source and is not used. Pose deltas would require accumulated translation and composed rotation rather than averaging per-sample deltas.

The seventh action is the event-driven normalized gripper request, `0` closed and `1` open. Latch its latest value at the interval start. Keep the physical knuckle angle in radians in state slot 17. Trim the prefix before the first recorded gripper request instead of inventing its initial value.

Fit one fixed 100 ms grid per episode after each observation's completed-record timestamp. Require the latest available tuple at every grid start to be the corresponding original tuple, all modality receipts to precede that start, and the oldest receipt to be at most 100 ms old. Reject skipped or repeated tuples, reordered clocks, uncovered request intervals, and gripper events that would merge or disappear. Trim incomplete terminal intervals.

The previous datasets already declare 10 Hz. Their 15,366 labels integrate requests over camera intervals of 67.344–102.902 ms, with a 101.245 ms median, and then assign uniform 100 ms dataset timestamps. The largest per-episode change in request integral from that retiming is 0.0433713 normalized request-seconds. In 21,228 of all 21,234 raw tuples, at least one state receipt follows the old camera-based anchor, by up to 33.761892 ms. The audit field `old_labels_with_future_state` counts raw tuples, including rows outside the old exported spans. This conversion fixes those alignment and interval issues; it does not demonstrate a tenfold training-frequency error or resolve the observed weak movement by itself.

## Dataset And Validation

The separate output root is:

`/data1/Muttaqien/datasets/fr3_peg_request_10hz_phase1_v1`

| Split | Original episode identities | Frames | Full ten-action horizon starts |
| --- | ---: | ---: | ---: |
| `train` | 80 | 12,240 | 11,520 |
| `validation` | 20 | 3,065 | 2,885 |

The exact episode assignment comes from the existing `fr3_peg_request_lerobot_{train,validation}_v1` provenance. Total retained length is 15,305 frames, 61 fewer than the old export. Individual episodes contain 124–183 frames, or 12.4–18.3 seconds. Unknown-gripper prefixes and incomplete tail intervals account for trimming. Retained observations are contiguous and preserve the measured spatial trajectory; no positions, orientations, joints, or images are interpolated.

The evidence accompanies the data. A compact repository copy is [fr3_frequency_phase1_evidence.json](fr3_frequency_phase1_evidence.json):

| Artifact | Purpose |
| --- | --- |
| `audit.json` | Source timing distributions, per-episode bounds, source SHA-256 values, final export checks, completion flag |
| `resampled/episode_*.jsonl` | Each exact interval, original frame identity, receipts, completed-record timestamp, state and action |
| `{train,validation}/frequency_provenance.json` | Original episode to exported episode mapping |
| `{train,validation}/meta/info.json` | LeRobot 10 Hz metadata and feature contract |
| `norm/norm_stats.json` | Fresh 32D padded statistics from float32 training rows only |
| `norm/provenance.json` | The 80 contributing training episodes; zero validation samples |

Every exported state/action and every RGB pixel is read back from Parquet and compared with its source. Original BGR data is converted once to lossless RGB. The two views remain `image` and `extra_view_image`; source frame index and monotonic grid time are also stored as int64 features. Boundary validation checks episode lengths and terminal `done`; terminal `is_success` reflects each original operator-reviewed successful episode, not an inferred insertion timestamp.

The oldest retained modality is at most 75.606503 ms old. All 200 gripper events survive, with 0–99.194223 ms quantization delay. The maximum request-integral discrepancy is below `1.03e-14` in float64 and `3.99e-8` after float32 export. Retained TCP positions are identical to the original positions. Original recording metadata is hashed before and after conversion; source recordings and previous datasets remain unchanged.

Offline verification passes 25 unit tests across the converter, existing alignment, normalization, export, and policy contracts. A model-free smoke check through `build_official_openpi_sft_dataloader` passes for both splits: state `(1,32)`, actions `(1,10,32)`, each active RGB view `(1,3,224,224)`, and the full-horizon counts above. Independent statistics recomputed from exported training rows match the saved normalization. Both EN/ZH Sphinx builds and scoped pre-commit checks pass.

## Reproduce The Offline Conversion

Run the source audit in a new writable directory from the repository root:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python \
  -m toolkits.realworld_check.fr3_frequency_dataset \
  --output /tmp/fr3_frequency_audit_new --audit-only
```

This reads the raw source and both reference splits, validates timing and label semantics, and writes audit evidence and resampled JSONL. Omit `--audit-only` and choose another new output directory to export LeRobot data, read back every pixel and label, and compute training-only statistics. Existing or overlapping output directories are refused. The tool never initializes a robot or a training worker.

## Proposed SFT Configuration

Review [`realworld_sft_openpi_fr3_peg_request_10hz_phase1.yaml`](../../../examples/sft/config/realworld_sft_openpi_fr3_peg_request_10hz_phase1.yaml) before authorizing training. It selects the new train, validation, and norm paths while retaining the existing π₀.₅ base model, 19D measured state padded to 32, two RGB cameras, seven normalized requests padded to 32, ten-action horizon, seed, optimizer, and proposed 3,000-step schedule. Validation is proposed every 500 steps. A ten-action horizon covers one second; only full horizons within an episode enter the official OpenPI loader.

The recorded task prompt is **`FR3 peg insertion`**, supplied by `prompt_from_task=True`. Preserve it for this dataset. The existing native evaluation configuration uses a different prompt; reconcile that separately before comparing future checkpoints. This preparation leaves the native evaluation configuration intact.

The frequency contract is 100 Hz raw requests and raw state where actually recorded, 10 Hz training targets and policy actions, and 100 Hz execution refresh of held policy requests. A refresh does not multiply an action by ten or introduce a 0.15 scale. The current recordings support reliable 10 Hz training preparation under the measured host-receipt bounds. Claims requiring calibrated exposure timing, retrospective 100 Hz state, or physical execution equivalence need separate evidence. No SFT run or physical evaluation is started in Phase 1.
