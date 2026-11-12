# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Non-actuating diagnostics for normalized SpaceMouse REQUEST transitions.

Requires all 10-step validation windows in exported segment order. Uses the
first predicted action of each window exactly once. Segment boundaries must
never be treated as contiguous time or as robot controller telemetry.
"""

from collections import OrderedDict
import numpy as np

AXES = ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z")


def _div(a, b):
    return float(a / b) if b else None


def _f1(tp, fp, fn):
    return _div(2 * tp, 2 * tp + fp + fn)


def _events(states, direction):
    s = np.asarray(states, dtype=bool)
    return np.flatnonzero((s[1:] != s[:-1]) & (s[1:] == direction)) + 1


def match_events(truth, prediction, tolerance=2):
    """Greedy one-to-one event matching within each contiguous segment."""
    available = set(range(len(prediction)))
    signed_lags = []
    for t in truth:
        candidates = [(abs(int(prediction[i]) - int(t)), i) for i in available
                      if abs(int(prediction[i]) - int(t)) <= tolerance]
        if candidates:
            _, j = min(candidates)
            signed_lags.append(int(prediction[j]) - int(t))
            available.remove(j)
    return len(signed_lags), signed_lags


def transition_score(sequences, *, direction, tolerance=2):
    actual = predicted = matched = 0
    lags = []
    for true_state, pred_state in sequences:
        a, p = _events(true_state, direction), _events(pred_state, direction)
        count, offset = match_events(a, p, tolerance)
        actual += len(a)
        predicted += len(p)
        matched += count
        lags.extend(offset)
    return {
        "true_events": int(actual),
        "predicted_events": int(predicted),
        "matched_events": int(matched),
        "precision": _div(matched, predicted),
        "recall": _div(matched, actual),
        "f1": _f1(matched, predicted - matched, actual - matched),
        "mean_abs_lag_frames": float(np.mean(np.abs(lags))) if lags else None,
        "mean_signed_lag_frames": float(np.mean(lags)) if lags else None,
    }


def _axis_scores(p, y, *, threshold):
    output = {}
    for j, name in enumerate(AXES):
        pa, ya = p[:, j], y[:, j]
        active = np.abs(ya) >= threshold
        predicted_active = np.abs(pa) >= threshold
        tp = int(np.sum(active & predicted_active))
        fp = int(np.sum(~active & predicted_active))
        fn = int(np.sum(active & ~predicted_active))
        output[name] = {
            "first_step_rmse": float(np.sqrt(np.mean((pa - ya) ** 2))),
            "zero_baseline_rmse": float(np.sqrt(np.mean(ya ** 2))),
            "true_active_steps": int(np.sum(active)),
            "activity_f1": _f1(tp, fp, fn),
            "correct_sign_given_true_active": (
                float(np.mean(np.sign(pa[active]) == np.sign(ya[active])))
                if np.any(active) else None
            ),
            "true_active_rmse": (
                float(np.sqrt(np.mean((pa[active] - ya[active]) ** 2)))
                if np.any(active) else None
            ),
            "true_active_zero_baseline_rmse": (
                float(np.sqrt(np.mean(ya[active] ** 2)))
                if np.any(active) else None
            ),
        }
    return output


def analyze(predicted, target, segments, *, threshold=0.1, tolerance=2):
    p, y = np.asarray(predicted, dtype=float), np.asarray(target, dtype=float)
    if (p.shape != y.shape or p.ndim != 3 or p.shape[1:] != (10, 7)
            or not np.isfinite(p).all() or not np.isfinite(y).all()
            or not 0 < threshold < 1 or not 0 <= tolerance <= 10):
        raise ValueError("Expected finite full-window arrays (N,10,7), valid settings")
    if not segments:
        raise ValueError("Missing contiguous source segments")
    p0, y0 = p[:, 0, :], y[:, 0, :]
    ranges = []
    episode_ranges = OrderedDict()
    cursor = 0
    for seg in segments:
        name, start, stop = seg.get("episode"), seg.get("start"), seg.get("stop")
        if (not isinstance(name, str) or not name.startswith("episode_")
                or type(start) is not int or type(stop) is not int
                or stop - start < 10):
            raise ValueError("Invalid episode/segment provenance")
        length = stop - start - 9
        ranges.append((cursor, cursor + length))
        episode_ranges.setdefault(name, []).append((cursor, cursor + length))
        cursor += length
    if cursor != len(y0):
        raise ValueError("All windows required in original strict-window order")

    scores = _axis_scores(p0, y0, threshold=threshold)
    for j, name in enumerate(AXES):
        sub = [(np.abs(y0[a:b, j]) >= threshold,
                np.abs(p0[a:b, j]) >= threshold) for a, b in ranges]
        scores[name]["onset"] = transition_score(
            sub, direction=True, tolerance=tolerance
        )
        scores[name]["offset"] = transition_score(
            sub, direction=False, tolerance=tolerance
        )

    grip_true, grip_pred = y0[:, 6] >= 0.5, p0[:, 6] >= 0.5
    grip_segments = [(grip_true[a:b], grip_pred[a:b]) for a, b in ranges]
    episodes = {}
    for name, parts in episode_ranges.items():
        indices = np.concatenate([np.arange(a, b) for a, b in parts])
        ep_p, ep_y = p0[indices], y0[indices]
        episodes[name] = {
            "distinct_start_steps": len(indices),
            "arm_6d_rmse": float(np.sqrt(np.mean(
                (ep_p[:, :6] - ep_y[:, :6]) ** 2
            ))),
            "rotation": {
                key: _axis_scores(ep_p, ep_y, threshold=threshold)[key]
                for key in AXES[3:]
            },
            "gripper_first_step_accuracy": float(np.mean(
                (ep_p[:, 6] >= 0.5) == (ep_y[:, 6] >= 0.5)
            )),
        }

    return {
        "schema": "fr3_request_transition_diagnostic_v1",
        "requests_only": True,
        "physical_execution_approved": False,
        "heldout_source_episodes": len(episode_ranges),
        "contiguous_segments": len(ranges),
        "distinct_scored_steps": len(p0),
        "tail_frames_omitted": 9 * len(ranges),
        "first_action_per_window_only": True,
        "activity_threshold_abs_request": threshold,
        "event_tolerance_frames": tolerance,
        "nominal_frame_period_s": 0.1,
        "axes": scores,
        "gripper": {
            "first_step_accuracy": float(np.mean(grip_true == grip_pred)),
            "first_step_majority_baseline": float(max(
                np.mean(grip_true), 1 - np.mean(grip_true)
            )),
            "open_transition": transition_score(
                grip_segments, direction=True, tolerance=tolerance
            ),
            "close_transition": transition_score(
                grip_segments, direction=False, tolerance=tolerance
            ),
        },
        "per_episode": episodes,
        "limitations": [
            "No physical peg-insertion success claim.",
            "Only horizon step 0 at each valid start; last 9 steps of each segment omitted.",
            "Onsets or offsets already in progress at segment starts are left-censored.",
            "Event tolerance is nominal samples, not calibrated execution latency.",
            "SpaceMouse request is not verified controller acceptance or safe actuation.",
        ],
    }
