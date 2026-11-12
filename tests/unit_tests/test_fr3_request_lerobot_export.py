"""Offline LeRobot-request exporter preflight tests."""
import hashlib
import json
import pytest

from toolkits.realworld_check.fr3_request_lerobot_export import preflight


def _fixture(tmp_path):
    raw = tmp_path / "raw"
    candidate = tmp_path / "candidate"
    (candidate / "episodes").mkdir(parents=True)
    reports, manifests = [], []
    for i, split in ((1, "train"), (2, "validation")):
        name = f"episode_{i:04d}"
        ep = raw / name
        for role in ("base", "wrist"):
            (ep / "frames" / role).mkdir(parents=True)
        manifest = {
            "task": "peg insertion", "sample_rate_hz_requested": 10,
            "camera_profiles": {
                "base": {"resolution": [2, 2]},
                "wrist": {"resolution": [2, 2]},
            },
        }
        (ep / "manifest.json").write_text(json.dumps(manifest))
        (ep / "episode_review.json").write_text(json.dumps({
            "operator_outcome": "success", "capture_status": "raw_recorded",
        }))
        (ep / "operator_commands.jsonl").write_text("original")
        rows = []
        for n in range(10):
            refs = {}
            for role in ("base", "wrist"):
                refs[role] = f"frames/{role}/{n:06d}.npy"
                (ep / refs[role]).write_bytes(b"reference")
            rows.append({
                "frame_index": n, "state19_candidate": [0] * 19,
                "candidate_complete": True,
                "candidate_request_action_7d": [0.2] * 6 + [1.0],
                "images_bgr8_npy": refs,
            })
        (candidate / "episodes" / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        reports.append({
            "episode": name,
            "raw_manifest_sha256": hashlib.sha256(
                (ep / "manifest.json").read_bytes()).hexdigest(),
            "recorded_requests_sha256": hashlib.sha256(
                (ep / "operator_commands.jsonl").read_bytes()).hexdigest(),
        })
        manifests.append({
            "episode": name, "split": split, "transitions": 10,
            "valid_spans_end_exclusive": [[0, 10]],
            "counts": {"complete_7d_request": 10},
        })
    (candidate / "report.json").write_text(json.dumps({
        "schema": "fr3_spacemouse_request_alignment_candidate_v1",
        "source_root": str(raw), "episode_reports": reports,
        "sft_ready": False, "lerobot_exported": False,
    }))
    quality = tmp_path / "quality.json"
    quality.write_text(json.dumps({
        "schema": "fr3_spacemouse_request_quality_v1",
        "source_candidate_root": str(candidate.resolve()),
        "horizon": 10, "gripper_unknown_imputed": False, "sft_ready": False,
        "episode_manifests": manifests,
        "train": {"complete_7d_transitions": 10},
        "validation": {"complete_7d_transitions": 10},
    }))
    return candidate, quality, raw


def test_train_validation_preflight_isolated(tmp_path):
    candidate, quality, _ = _fixture(tmp_path)
    train = preflight(candidate, quality, "train")
    val = preflight(candidate, quality, "validation")
    assert train["action_labels"] == val["action_labels"] == 10
    assert train["segments"][0]["episode"] != val["segments"][0]["episode"]
    assert train["sft_ready"] is False


def test_modified_source_commands_rejected(tmp_path):
    candidate, quality, raw = _fixture(tmp_path)
    (raw / "episode_0001" / "operator_commands.jsonl").write_text("modified")
    with pytest.raises(ValueError, match="operator requests changed"):
        preflight(candidate, quality, "train")


def test_unknown_action_rejected(tmp_path):
    candidate, quality, _ = _fixture(tmp_path)
    file = candidate / "episodes" / "episode_0001.jsonl"
    rows = [json.loads(x) for x in file.read_text().splitlines()]
    rows[0]["candidate_request_action_7d"] = None
    rows[0]["candidate_complete"] = False
    file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="invalid request label"):
        preflight(candidate, quality, "train")
