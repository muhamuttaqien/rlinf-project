# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Read-only temporal diagnostics for FR3 normalized SpaceMouse requests.

Compare two independently seeded pi0.5 samples against one constant-noise
sample across the SAME ordered held-out observation windows. Never changes
the model, dataset, or robot. Events use horizon step 0 exactly once.
"""

from __future__ import annotations

import numpy as np

from toolkits.realworld_check.fr3_request_transition_diagnostic import AXES, analyze


def _ranges(segments, count):
    if not segments:
        raise ValueError("No exported validation segments")
    result = []
    cursor = 0
    for segment in segments:
        name, first, stop = (
            segment.get("episode"), segment.get("start"), segment.get("stop")
        )
        if (not isinstance(name, str) or not name.startswith("episode_")
                or type(first) is not int or type(stop) is not int
                or stop - first < 10):
            raise ValueError("Invalid source segment provenance")
        length = stop - first - 9
        result.append((name, cursor, cursor + length))
        cursor += length
    if cursor != count:
        raise ValueError("Expected all strict windows in segment order")
    return result


def _overlap_check(labels, ranges, *, atol=1e-4):
    """Consecutive target action windows should refer to identical future steps."""
    mismatch = 0
    compared = 0
    largest = np.zeros(7)
    for _, start, stop in ranges:
        if stop - start < 2:
            continue
        a = labels[start:stop-1, 1:, :]
        b = labels[start+1:stop, :-1, :]
        delta = np.abs(a - b)
        largest = np.maximum(largest, np.max(delta, axis=(0, 1)))
        mismatch += int(np.sum(np.any(delta > atol, axis=-1)))
        compared += int(delta.shape[0] * delta.shape[1])
    return {
        "compared_future_action_vectors": compared,
        "mismatched_vectors_at_tolerance": mismatch,
        "tolerance": atol,
        "max_abs_difference_by_channel": {
            name: float(largest[k]) for k, name in enumerate((*AXES, "gripper_open"))
        },
        "all_overlaps_consistent": mismatch == 0,
    }


def _single_sample_activity_bursts(state):
    """Count one-frame 'active' runs. These are NOT automatically noisy labels."""
    v = np.asarray(state, dtype=np.int8)
    d = np.diff(np.pad(v, (1, 1), mode="constant"))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int(np.sum((ends - starts) == 1))


def _activity_dynamics(x, ranges, threshold):
    values = x[:, 0, :]
    result = {}
    for k, name in enumerate((*AXES, "gripper_open")):
        transitions = one_step_bursts = 0
        variations = []
        for _, first, last in ranges:
            s = (values[first:last, k] >= 0.5 if k == 6
                 else np.abs(values[first:last, k]) >= threshold)
            transitions += int(np.sum(s[1:] != s[:-1]))
            one_step_bursts += _single_sample_activity_bursts(s)
            variations.extend(np.abs(np.diff(values[first:last, k])).tolist())
        result[name] = {
            "activity_state_transitions": transitions,
            "single_step_active_runs": one_step_bursts,
            "mean_abs_adjacent_command_change": (
                float(np.mean(variations)) if variations else None
            ),
        }
    return result


def _compact_transition(x, target, segments, threshold, tolerance):
    out = analyze(x, target, segments, threshold=threshold, tolerance=tolerance)
    return {
        "axes": {
            name: {
                "rmse": out["axes"][name]["first_step_rmse"],
                "zero_baseline_rmse": out["axes"][name]["zero_baseline_rmse"],
                "activity_f1": out["axes"][name]["activity_f1"],
                "sign_accuracy_when_target_active":
                    out["axes"][name]["correct_sign_given_true_active"],
                "onset": out["axes"][name]["onset"],
                "offset": out["axes"][name]["offset"],
            }
            for name in AXES
        },
        "gripper": out["gripper"],
        "worst_episodes_by_6d_rmse": sorted(
            (
                {"episode": ep, "arm_6d_rmse": row["arm_6d_rmse"]}
                for ep, row in out["per_episode"].items()
            ),
            key=lambda row: row["arm_6d_rmse"], reverse=True
        )[:5],
    }


def _sensitivity(a, b, threshold):
    """Two sampled outputs for the SAME observation, one first-step action each."""
    u, v = a[:, 0, :], b[:, 0, :]
    result = {}
    for j, key in enumerate((*AXES, "gripper_open")):
        state1 = (u[:, j] >= 0.5 if j == 6 else np.abs(u[:, j]) >= threshold)
        state2 = (v[:, j] >= 0.5 if j == 6 else np.abs(v[:, j]) >= threshold)
        result[key] = {
            "first_step_inter_sample_rmse": float(
                np.sqrt(np.mean((u[:, j] - v[:, j]) ** 2))
            ),
            "activity_disagreement_fraction": float(np.mean(state1 != state2)),
        }
    return result


def summarize(independent_a, independent_b, fixed_noise, target, segments, *,
              threshold=0.1, tolerance=2, independent_seeds=(0, 1),
              fixed_noise_seed=1001):
    """Separate request-label dynamics, random-noise sensitivity, and residual errors.

    Shared-noise controls *only* the sampled initial flow noise. It does not
    isolate every source of model error, prove label corruption, or emulate
    actual policy execution.
    """
    arrays = [
        np.asarray(a, dtype=np.float64)
        for a in (independent_a, independent_b, fixed_noise, target)
    ]
    a, b, fixed, y = arrays
    if (a.ndim != 3 or a.shape[1:] != (10, 7)
            or any(x.shape != a.shape or not np.isfinite(x).all() for x in arrays)):
        raise ValueError("Expected same finite (N,10,7) arrays")
    if (not 0 < threshold < 1 or type(tolerance) is not int
            or not 0 <= tolerance <= 10):
        raise ValueError("Invalid activity threshold or transition tolerance")
    if (len(independent_seeds) != 2 or independent_seeds[0] == independent_seeds[1]
            or fixed_noise_seed in independent_seeds):
        raise ValueError("Independent and shared-noise seeds must differ")
    ranges = _ranges(segments, len(a))

    return {
        "schema": "fr3_request_temporal_consistency_v1",
        "scope": "offline held-out request prediction, no controller or robot actions",
        "heldout_source_episodes": len(set(name for name, _, _ in ranges)),
        "exported_contiguous_segments": len(ranges),
        "distinct_window_starts": len(a),
        "scored": "first action at each window start; 9 tail steps per segment excluded",
        "independent_seeds": list(independent_seeds),
        "fixed_noise_seed": fixed_noise_seed,
        "activity_threshold": threshold,
        "transition_tolerance_frames": tolerance,
        "nominal_frame_period_s": 0.1,
        "source_label_self_consistency": _overlap_check(y, ranges),
        "first_step_activity_dynamics": {
            "target": _activity_dynamics(y, ranges, threshold),
            "independent_seed_a": _activity_dynamics(a, ranges, threshold),
            "independent_seed_b": _activity_dynamics(b, ranges, threshold),
            "shared_fixed_noise": _activity_dynamics(fixed, ranges, threshold),
        },
        "sampling_sensitivity": {
            "two_independent_seeds_same_observations": _sensitivity(a, b, threshold),
            "independent_a_vs_shared_noise_same_observations": _sensitivity(a, fixed, threshold),
        },
        "prediction_performance": {
            "independent_seed_a": _compact_transition(a, y, segments, threshold, tolerance),
            "independent_seed_b": _compact_transition(b, y, segments, threshold, tolerance),
            "shared_fixed_noise": _compact_transition(fixed, y, segments, threshold, tolerance),
        },
        "interpretation_limits": [
            "Stable target window overlaps check labeling consistency, NOT physical timestamp alignment.",
            "Single-step activity bursts in recorded requests are not automatically label noise.",
            "Shared fixed initial noise is a controlled sampling ablation, not an executable strategy.",
            "Two independent samples are insufficient to estimate all stochastic behavior.",
            "The task is not tested in closed loop; insertion success remains unknown.",
            "Physical robot execution is not approved.",
        ],
        "physical_execution_approved": False,
    }
