# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Evidence integrity/review mechanics using temporary synthetic fixtures only."""

import json
import shutil
from datetime import datetime, timezone

import pytest
from robot_mocks.mapping_calibration import FakeConsole, FakeRuntime, deployment_file

from toolkits.realworld_check.fr3_mapping_calibration import (
    Blocked,
    Plan,
    execute_case,
    load_deployment,
)
from toolkits.realworld_check.fr3_mapping_evidence import (
    analyze_session,
    apply_review,
    audit_record,
    file_record,
    finalize_record,
    seal_session,
    validate_review,
    verify_session,
    write_json,
)
from toolkits.realworld_check.fr3_pi05_request_adapter import ARM_AXES, MAPPING_CHECKS


def pending_record(path):
    write_json(
        path,
        {
            "schema": "fr3_pi05_mapping_verification_v1",
            "mapping_mode": "identity_verified",
            "arm_axes": list(ARM_AXES),
            "policy_period_ns": 100_000_000,
            "publish_period_ns": 10_000_000,
            "operator": None,
            "verified_at_utc": None,
            "checks": {k: {"passed": False} for k in MAPPING_CHECKS},
        },
    )
    return path


def schema_session(
    tmp_path,
    case="linear_x_positive",
    *,
    ros_schema=False,
    stop_after_ns=None,
    clock_start_ns=None,
):
    """Create a synthetic ROS-schema fixture, never physical evidence.

    Positive review-mechanics tests need the real schema's backend discriminator.
    These fixtures exist only in pytest temporary directories and are not exports.
    """
    path = deployment_file(tmp_path / "deployment")
    runtime = FakeRuntime(load_deployment(path), gripper=case.startswith("gripper_"))
    if clock_start_ns is not None:
        runtime.clock = clock_start_ns
    console = FakeConsole(runtime, stop_after_ns=stop_after_ns)
    output = tmp_path / "SYNTHETIC_SCHEMA_TEST_ONLY"
    execute_case(runtime, Plan(case), path, output, console)
    if ros_schema:
        for filename in ("session.json", "summary.json"):
            target = output / filename
            value = json.loads(target.read_text())
            value["backend"] = "ros"
            target.write_text(json.dumps(value))
        (output / "manifest.json").unlink()
        seal_session(output)
    return output, console


def review_file(session, path, check="linear_x_positive", verdict="pass"):
    analysis = analyze_session(session)
    attachment = path.parent / "SYNTHETIC_OPERATOR_NOTES.txt"
    attachment.write_text(
        "Unit-test fixture only; NOT physical measurements or human approval.\n"
    )
    write_json(
        path,
        {
            "schema": "fr3_mapping_operator_review_v1",
            "operator": "SYNTHETIC UNIT TEST",
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            "check": check,
            "verdict": verdict,
            "session_manifest_sha256": analysis["session_manifest_sha256"],
            "scope_sha256": analysis["scope_sha256"],
            "measurements": {"fixture_only": True},
            "criteria": {"fixture_only": True},
            "notes": "Synthetic review mechanics only",
            "attachments": [file_record(attachment, path.parent)],
        },
    )
    return path


@pytest.fixture
def session(tmp_path):
    return schema_session(tmp_path)[0]


def test_analysis_never_turns_fake_motion_or_counters_into_a_physical_pass(session):
    report = analyze_session(session)
    assert report["backend"] == "fake"
    assert report["physical_checks_passed"] == []
    assert report["controller_counter_deltas"]["moving_rx"] > 0
    assert report["frame_motion"]["flange"]["max_translation_m"] > 0
    assert report["controller_receipt_timing"]["available"] is False


@pytest.mark.parametrize(
    "filename", ["events.jsonl", "telemetry.jsonl", "summary.json", "session.json"]
)
def test_any_changed_stream_or_header_invalidates_sealed_evidence(session, filename):
    path = session / filename
    path.write_text(path.read_text() + " ")
    with pytest.raises(Blocked, match="mismatch"):
        verify_session(session)


def test_added_unsealed_file_is_rejected(session):
    (session / "unreviewed.txt").write_text("not sealed")
    with pytest.raises(Blocked, match="complete sealed manifest"):
        verify_session(session)


def test_symlink_escape_is_rejected(session, tmp_path):
    target = tmp_path / "external"
    target.write_text("external")
    (session / "escape").symlink_to(target)
    (session / "manifest.json").unlink()
    seal_session(session)
    with pytest.raises(Blocked, match="inside its sealed bundle"):
        verify_session(session)


def test_fake_hardware_cannot_be_promoted_by_an_operator_review(session, tmp_path):
    review = review_file(session, tmp_path / "review.json")
    with pytest.raises(Blocked, match="Fake/offline"):
        validate_review(review, session)


@pytest.mark.parametrize(
    "field",
    [
        "operator",
        "notes",
        "measurements",
        "criteria",
        "attachments",
        "session_manifest_sha256",
        "scope_sha256",
        "reviewed_at_utc",
    ],
)
def test_incomplete_or_unbound_operator_review_is_rejected(tmp_path, field):
    session, _ = schema_session(tmp_path, ros_schema=True)
    path = review_file(session, tmp_path / "review.json")
    review = json.loads(path.read_text())
    review[field] = None
    path.write_text(json.dumps(review))
    with pytest.raises(Blocked):
        validate_review(path, session)


def test_wrong_signed_axis_cannot_reuse_a_different_direction(tmp_path):
    session, _ = schema_session(tmp_path, ros_schema=True)
    review = review_file(session, tmp_path / "review.json", "linear_x_negative")
    with pytest.raises(Blocked, match="exact case"):
        validate_review(review, session)


def test_completed_pulse_without_operator_stop_cannot_pass_stop_check(tmp_path):
    session, _ = schema_session(tmp_path, ros_schema=True)
    review = review_file(session, tmp_path / "review.json", "neutral_stop")
    with pytest.raises(Blocked, match="operator stop during"):
        validate_review(review, session)


def test_one_operator_stopped_trace_can_support_direction_and_stop_reviews(tmp_path):
    session, _ = schema_session(tmp_path, ros_schema=True, stop_after_ns=350_000_000)
    for i, check in enumerate(("linear_x_positive", "neutral_stop")):
        review = review_file(session, tmp_path / f"review_{i}.json", check)
        result, analysis = validate_review(review, session)
        assert result["check"] == check
        assert analysis["status"] == "operator_stop"


def test_partial_review_copies_portable_evidence_without_final_approval(tmp_path):
    session, console = schema_session(tmp_path / "source", ros_schema=True)
    record = pending_record(tmp_path / "pending.json")
    review = review_file(session, tmp_path / "review.json")
    output = tmp_path / "partial.json"
    apply_review(record, session, review, output, console)
    result = json.loads(output.read_text())
    assert result["checks"]["linear_x_positive"]["passed"] is True
    assert result["operator"] is None and result["verified_at_utc"] is None
    assert result["physical_execution_approved"] is False
    assert audit_record(output)["passed"] == 1
    # The record and bundle work after the original source capture disappears.
    shutil.rmtree(tmp_path / "source")
    review.unlink()
    (tmp_path / "SYNTHETIC_OPERATOR_NOTES.txt").unlink()
    assert audit_record(output)["passed"] == 1
    with pytest.raises(Blocked, match="17 physical reviews"):
        finalize_record(output, tmp_path / "final.json", "SYNTHETIC UNIT TEST", console)


def test_changed_nested_measurement_invalidates_record_even_when_root_proof_hash_is_unchanged(
    tmp_path,
):
    session, console = schema_session(tmp_path / "source", ros_schema=True)
    record = pending_record(tmp_path / "pending.json")
    review = review_file(session, tmp_path / "review.json")
    output = tmp_path / "partial.json"
    apply_review(record, session, review, output, console)
    target = output.with_suffix(".evidence") / "session/telemetry.jsonl"
    target.write_text(target.read_text() + " ")
    with pytest.raises(Blocked, match="mismatch"):
        audit_record(output)


def test_record_binary_fingerprint_cannot_differ_from_measured_scope(tmp_path):
    session, console = schema_session(tmp_path / "source", ros_schema=True)
    record = pending_record(tmp_path / "pending.json")
    review = review_file(session, tmp_path / "review.json")
    output = tmp_path / "partial.json"
    apply_review(record, session, review, output, console)
    raw = json.loads(output.read_text())
    raw["controller_binary_sha256"] = "0" * 64
    output.write_text(json.dumps(raw))
    with pytest.raises(Blocked, match="installation/frame"):
        audit_record(output)


def test_inconclusive_evidence_is_also_verified_transitively(tmp_path):
    session, console = schema_session(tmp_path / "source", ros_schema=True)
    record = pending_record(tmp_path / "pending.json")
    review = review_file(session, tmp_path / "review.json", verdict="inconclusive")
    output = tmp_path / "partial.json"
    apply_review(record, session, review, output, console)
    assert audit_record(output)["passed"] == 0
    target = output.with_suffix(".evidence") / "session/telemetry.jsonl"
    target.write_text(target.read_text() + " ")
    with pytest.raises(Blocked, match="mismatch"):
        audit_record(output)


def test_existing_review_output_is_never_overwritten(tmp_path):
    session, console = schema_session(tmp_path / "source", ros_schema=True)
    record = pending_record(tmp_path / "pending.json")
    review = review_file(session, tmp_path / "review.json")
    output = tmp_path / "partial.json"
    output.write_text("existing result")
    with pytest.raises(Blocked, match="new path"):
        apply_review(record, session, review, output, console)
    assert output.read_text() == "existing result"


def test_gripper_pass_requires_a_real_counterpart_session(tmp_path):
    session, _ = schema_session(tmp_path, "gripper_open", ros_schema=True)
    review = review_file(session, tmp_path / "review.json", "gripper_open_close")
    with pytest.raises(Blocked, match="counterpart_session_manifest"):
        validate_review(review, session)


def test_pending_record_reports_zero_passes_and_cannot_finalize(tmp_path):
    record = pending_record(tmp_path / "pending.json")
    assert audit_record(record)["passed"] == 0
    with pytest.raises(Blocked, match="17 physical reviews"):
        finalize_record(record, tmp_path / "final.json", "SYNTHETIC UNIT TEST", None)


def test_complete_review_chain_finalizes_only_after_final_consent(tmp_path):
    sessions = {}
    console = None
    for i, case in enumerate((*MAPPING_CHECKS[:12], "gripper_open", "gripper_close")):
        sessions[case], console = schema_session(
            tmp_path / case,
            case,
            ros_schema=True,
            stop_after_ns=350_000_000 if case == "linear_x_positive" else None,
            clock_start_ns=1_000_000_000 + i * 10_000_000_000,
        )
    record = pending_record(tmp_path / "pending.json")
    for i, check in enumerate(MAPPING_CHECKS):
        case = (
            check
            if check in sessions
            else "gripper_open"
            if check == "gripper_open_close"
            else "linear_x_positive"
        )
        session = sessions[case]
        review = review_file(session, tmp_path / f"review_{i}.json", check)
        if check == "gripper_open_close":
            raw = json.loads(review.read_text())
            raw["counterpart_session_manifest"] = file_record(
                sessions["gripper_close"] / "manifest.json", tmp_path
            )
            review.write_text(json.dumps(raw))
        output = tmp_path / f"partial_{i}.json"
        apply_review(record, session, review, output, console)
        record = output
    assert audit_record(record)["passed"] == 17
    assert not audit_record(record)["physical_execution_approved"]
    final = tmp_path / "final.json"
    finalize_record(record, final, "SYNTHETIC UNIT TEST", console)
    assert console.tokens[-1].startswith("APPROVE MAPPING ")
    assert audit_record(final)["physical_execution_approved"]
    assert json.loads(final.read_text())["operator"] == "SYNTHETIC UNIT TEST"
