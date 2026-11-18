# π₀.₅ Gripper Prediction Validation

This report explains the offline correction for Episode 7's gripper validation abort. It traces the training labels and inverse normalization, records the observed overshoots, and describes the bounded correction and regression checks. [中文](pi05_gripper_validation.zh.md)

## Cause and Recorded Outcome

Episode 7 executed 30 policy actions with the 100 Hz arm refresh before the next chunk failed validation. Its first gripper prediction was `-0.05276369364422559`, outside the old `[-0.05, 1.05]` admission interval. The second prediction was lower, `-0.05716913694965839`; allowing only the first value would still reject this chunk. All six arm dimensions in the rejected chunk were within their existing limits.

The recorded outcome remains `aborted`, with `stop_reason=invalid_policy_action`, 30 policy actions, 456 arm publications, and a maximum publish gap of 11,727,168 ns. The rejected chunk was never executed. Hashes of all existing Episode 1–7 files were checked after offline replay and remained unchanged. Episode 3 has logs and observations but no `outcome.json`; this investigation preserves that state.

## Training Labels and Decoding

The seventh dimension estimates the last operator button request: `0=close`, `1=open`. It represents a latched request, not gripper width or measured knuckle position. `fr3_command_request_alignment.py` takes the last request before the observation window ends; `fr3_request_lerobot_export.py` retains that label. Unknown prefixes are excluded rather than inferred from measured position.

Reading the exported Parquet action columns confirmed the labels:

| Split | Action rows | Close (`0`) | Open (`1`) | Other values |
| --- | ---: | ---: | ---: | ---: |
| Training | 12,288 | 8,191 | 4,097 | 0 |
| Validation | 3,078 | 2,031 | 1,047 | 0 |

The saved training normalization has gripper `q01=0`, `q99=1`, mean `0.3334147135416667`, and standard deviation `0.4714332851375357`. `fr3_request_norm_stats.py` enforces binary labels. The rollout applies OpenPI `Unnormalize(..., use_quantiles=True)` to generated actions once, then selects the seven request dimensions. The installed OpenPI inverse is:

```text
decoded = (generated + 1) / 2 * (q99 - q01 + 1e-6) + q01
```

This affine transform imposes no output bound. A continuous model estimate can overshoot the binary target range; the saved values below show that this occurs routinely. The tiny normalization epsilon does not explain a roughly 0.057 overshoot. The abort came from applying the old admission bound to that estimate before interpreting the binary request.

## Observed Predictions and Admission Bound

Read-only inspection used existing saved predictions, with no model inference or robot access. The [evidence JSON](pi05_gripper_validation_evidence.json) records source paths, SHA-256 hashes, counts, normalization statistics, the rejected chunk, and replay results.

| Source | Predictions | Minimum | Maximum | Outside `[0, 1]` | Outside old bounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Offline validation shadow, 300 overlapping chunks | 3,000 | -0.02103265 | 1.04768977 | 741 (24.7%) | 0 |
| Live sensor shadow v2, saved predictions | 50 | -0.02261593 | 1.04127464 | 33 | 0 |
| Previously executed Episode 1–7 action logs | 430 | -0.04907386 | 1.04712043 | 239 | 0 |
| Episode 7 rejected chunk | 10 | -0.05716914 | 1.01893515 | 6 | 2 |

The first live-shadow directory has an empty predictions file and contributes no samples. The executed-action logs are biased toward predictions that passed the old gate; they cannot estimate how often rejected predictions occur. Overlapping offline horizons also do not represent independent trials.

The new admission interval is `[-0.06, 1.06]`. Its 0.06 overshoot allowance rounds the largest saved observed overshoot, `0.05716913694965839`, outward to the next 0.01 of the unit label span. This is a narrow empirical bound covering the complete failed chunk, not a guarantee about future model outputs. Values beyond it still abort; no general 0.1 allowance or unbounded clipping is introduced.

## Correction and Safety Contract

`validate_action_chunk()` first rejects nonnumeric/nonfinite values and checks the existing arm limits and the new gripper admission bound. For admitted values, it saturates only dimension seven to `[0, 1]` and returns a new chunk. Every arm value remains unchanged, including signed zero. A later invalid value rejects the entire chunk before any of its actions can be published.

The parent rollout now uses the returned validated chunk rather than discarding it. Accepted action records retain the original decoded row as `raw_predicted_7d`; `requested_7d` records the adapted request. Rejected chunks retain the existing raw representation in `invalid_policy_action.json` and the existing automatic `aborted` outcome. The executor independently validates incoming chunks through the same adapter.

The gripper filter keeps its `<=0.30` close threshold, `>=0.70` open threshold, three consecutive confirmations, and 0.8 s minimum goal interval. Saturating an admitted endpoint overshoot preserves its threshold classification. Arm scaling, mapping modes, 100 Hz transport refresh, controller code, freshness checks, displacement and duration limits, and stop handling retain their existing behavior. The trained checkpoint and normalization files were read only; no weights were loaded or changed.

## Offline Verification

The workstation checks used Python 3.11 on CPU, fake ROS message/publisher interfaces, and pipe-backed executor tests. They accessed no physical ROS endpoints. Run the component checks from the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/unit_tests/test_fr3_pi05_request_adapter.py \
  tests/unit_tests/test_fr3_pi05_ros_executor.py \
  tests/unit_tests/test_fr3_pi05_deployment_audit.py

.venv/bin/ruff check --preview \
  toolkits/realworld_check/fr3_pi05_request_adapter.py \
  toolkits/realworld_check/fr3_pi05_supervised_rollout.py \
  tests/unit_tests/test_fr3_pi05_request_adapter.py \
  tests/unit_tests/test_fr3_pi05_ros_executor.py

.venv/bin/ruff format --check \
  toolkits/realworld_check/fr3_pi05_request_adapter.py \
  toolkits/realworld_check/fr3_pi05_supervised_rollout.py \
  tests/unit_tests/test_fr3_pi05_request_adapter.py \
  tests/unit_tests/test_fr3_pi05_ros_executor.py

git diff --check
```

All 218 tests passed in both the working checkout and the isolated remote-tip worktree. Ruff lint, Ruff format, and `git diff --check` passed.

The repository docs build harness reported zero warnings for English and Chinese with its default autodoc-import filtering. The RST markup checker reported zero errors and 61 existing warnings in unchanged source pages. Local Markdown links and identical EN/ZH command blocks were checked separately because these experiment reports sit outside the Sphinx source trees.

The tests cover valid values, both overshoot directions and inclusive bounds, larger errors, NaN and both infinities in every dimension, unchanged arm values and limits, debounce thresholds and timing, and the saved Episode 7 chunk. Fake ROS tests also confirm that invalid late chunks fault and publish neutral arm commands without retry, and that displacement faults still stop execution when gripper saturation is active. Existing operator-stop, watchdog, exclusivity, scheduling, and archive-failure tests run alongside them.

The saved-prediction replay admitted all 3,490 inspected predictions, including the ten rejected Episode 7 predictions. The six arm values were byte-identical before and after adaptation; both existing arm mapping gains produced identical results. For the previously admitted data, replaying each ten-action chunk with fresh filters at 10 Hz preserved every open/close/hold decision. These checks establish the offline conversion contract, not a physical task outcome.

Work stops after offline verification. Any next physical episode requires separate operator approval and the existing per-run consent and safety gates. Episode 7 is not retried automatically.
