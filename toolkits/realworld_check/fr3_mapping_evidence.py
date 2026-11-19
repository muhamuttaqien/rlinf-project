# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Seal, inspect and explicitly review FR3 calibration evidence, without ROS."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from toolkits.realworld_check.fr3_mapping_calibration import (
    AXIS_CASES,
    DIAG_FIELDS,
    MAX_GAP_NS,
    PERIOD_NS,
    Blocked,
    Console,
    Plan,
    compose,
    deployment_scope,
    json_digest,
    message_pose,
    pose,
    rotation_vector,
    sha256,
)
from toolkits.realworld_check.fr3_pi05_request_adapter import (
    ARM_AXES,
    MAPPING_CHECKS,
    load_mapping_verification,
)


def write_json(path: Path, value: dict) -> None:
    """Create one JSON artifact exclusively; preserve previous attempts."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def file_record(path: Path, parent: Path) -> dict:
    """Describe file bytes using a relative path and SHA-256."""
    return {
        "path": os.path.relpath(path, parent),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def checked_file(parent: Path, item: dict, *, confined: bool = False) -> Path:
    """Check evidence hashes, rejecting symlinks or escape from sealed bundles."""
    raw = parent / item["path"]
    path = raw.resolve()
    if confined and (
        not path.is_relative_to(parent.resolve())
        or any(p.is_symlink() for p in (raw, *raw.parents) if p != parent.parent)
    ):
        raise Blocked(f"Evidence path must remain inside its sealed bundle: {raw}")
    if (
        not path.is_file()
        or sha256(path) != item["sha256"]
        or ("bytes" in item and path.stat().st_size != item["bytes"])
    ):
        raise Blocked(f"Evidence hash/size mismatch: {path}")
    return path


def seal_session(directory: Path) -> None:
    """Hash the closed streams, copied deployment inputs and terminal summary."""
    files = [
        file_record(path, directory)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    write_json(
        directory / "manifest.json",
        {"schema": "fr3_mapping_session_manifest_v1", "files": files},
    )


def verify_session(directory: Path) -> dict:
    """Verify storage integrity without interpreting measurements as approvals."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "fr3_mapping_session_manifest_v1":
        raise Blocked("Expected sealed calibration session manifest v1")
    listed = [x["path"] for x in manifest["files"]]
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "manifest.json"
    }
    if (
        len(listed) != len(set(listed))
        or set(listed) != actual
        or not {
            "session.json",
            "summary.json",
            "events.jsonl",
            "telemetry.jsonl",
        }.issubset(actual)
    ):
        raise Blocked("Session files differ from the complete sealed manifest")
    for item in manifest["files"]:
        checked_file(directory, item, confined=True)
    header = json.loads((directory / "session.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if (
        header.get("schema") != "fr3_mapping_calibration_session_v1"
        or json_digest(header["scope"]) != header["scope_sha256"]
    ):
        raise Blocked("Session scope identity mismatch")
    if deployment_scope(header["deployment"]) != header["scope"]:
        raise Blocked("Session deployment fields differ from its frozen scope")
    if (
        header["backend"] != summary["backend"]
        or header["plan"]["case"] != summary["case"]
    ):
        raise Blocked("Session backend/case differs from terminal summary")
    plan = Plan(**header["plan"])
    if asdict(plan) != header["plan"]:
        raise Blocked("Session plan does not match its declared contract")
    for field in ("controller_binary_sha256", "controller_config_sha256"):
        artifact = directory / "inputs" / header["scope"][field]
        if not artifact.is_file() or sha256(artifact) != header["scope"][field]:
            raise Blocked(f"Session lacks copied installation input: {field}")
    return {
        "manifest_sha256": sha256(directory / "manifest.json"),
        "header": header,
        "summary": summary,
    }


def read_rows(path: Path) -> list[dict]:
    """Read an intact JSONL stream; a truncated record invalidates the capture."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    stamps = [r.get("monotonic_ns", r.get("received_monotonic_ns")) for r in rows]
    if any(type(t) is not int or t <= 0 for t in stamps) or stamps != sorted(stamps):
        raise Blocked(f"Expected ordered positive host receipt/event times: {path}")
    return rows


def analyze_session(directory: Path) -> dict:
    """Report measured motion and timing, leaving all physical verdicts pending."""
    verified = verify_session(directory)
    header, terminal = verified["header"], verified["summary"]
    plan = Plan(**header["plan"])
    events, telemetry = (
        read_rows(directory / "events.jsonl"),
        read_rows(directory / "telemetry.jsonl"),
    )
    starts = [e for e in events if e["event"] == "case_started"]
    pubs = [e for e in events if e["event"] == "arm_publish"]
    stops = [e["monotonic_ns"] for e in events if e["event"] == "operator_stop"]
    issues = []
    if not starts or not pubs:
        issues.append("No executed case or publication trace")
    begin = starts[0]["monotonic_ns"] if starts else None
    for item in pubs:
        index = item["sample_index"]
        expected = (
            plan.sample(index)[0]
            if index is not None
            and type(index) is int
            and 0 <= index < plan.duration_ns // PERIOD_NS
            else [0.0] * 6
        )
        if item["request_6d"] != expected:
            issues.append("Published request differs from the case plan")
        if any(item["request_6d"]):
            if (
                begin is None
                or item["deadline_ns"] != begin + (index + 1) * PERIOD_NS
                or not begin + index * PERIOD_NS
                <= item["monotonic_ns"]
                < item["deadline_ns"]
            ):
                issues.append("Non-neutral publication outside its exact 100 ms slot")
            if stops and item["monotonic_ns"] >= min(stops):
                issues.append("Motion after processed operator stop")
    gaps = [b["monotonic_ns"] - a["monotonic_ns"] for a, b in zip(pubs, pubs[1:])]
    if gaps and max(gaps) > MAX_GAP_NS:
        issues.append("Publication gap exceeded 50 ms")
    goals = [e for e in events if e["event"] == "gripper_goal"]
    if (
        len(goals) > 1
        or (not plan.gripper and goals)
        or (plan.case == "gripper_hold" and goals)
    ):
        issues.append("Unexpected or repeated gripper goal")
    if stops and any(e["monotonic_ns"] >= min(stops) for e in goals):
        issues.append("Gripper goal after processed operator stop")
    samples = [e for e in events if e["event"] == "sample"]
    if terminal["status"] == "completed" and [e["index"] for e in samples] != list(
        range(plan.duration_ns // PERIOD_NS)
    ):
        issues.append("Completed case omitted a policy sample")
    if (
        plan.case in ("gripper_open", "gripper_close")
        and terminal["status"] == "completed"
    ):
        if (
            len(goals) != 1
            or goals[0]["goal"] != plan.case.split("_")[1]
            or len(samples) < 3
            or goals[0]["monotonic_ns"] < samples[2]["monotonic_ns"]
        ):
            issues.append("Gripper goal did not follow three policy confirmations")
    frames = {}
    if starts:
        origin = starts[0]["origin"]
        base = pose(header["deployment"]["base_to_pose"])
        for name, offset in (("flange", "ee_to_flange"), ("tcp", "ee_to_tcp")):
            values = [
                compose(
                    compose(base, message_pose(r["message"])),
                    pose(header["deployment"][offset]),
                )
                for r in telemetry
                if r["source"] == "pose" and r["received_monotonic_ns"] >= begin
            ]
            if values:
                frames[name] = {
                    "final_translation_m": [
                        values[-1][i] - origin[name][i] for i in range(3)
                    ],
                    "max_translation_m": max(
                        math.dist(origin[name][:3], p[:3]) for p in values
                    ),
                    "final_local_rotation_rad": rotation_vector(
                        tuple(origin[name]), values[-1]
                    ),
                    "max_rotation_rad": max(
                        math.dist(rotation_vector(tuple(origin[name]), p), (0, 0, 0))
                        for p in values
                    ),
                }
    diagnostics = [r["message"] for r in telemetry if r["source"] == "diagnostics"]
    counter_deltas = (
        {key: diagnostics[-1][key] - diagnostics[0][key] for key in DIAG_FIELDS[:9]}
        if len(diagnostics) >= 2
        else {}
    )
    if any(v < 0 for v in counter_deltas.values()):
        issues.append("Controller diagnostic counters reset during capture")
    result = {
        "schema": "fr3_mapping_calibration_analysis_v1",
        "session_manifest_sha256": verified["manifest_sha256"],
        "scope_sha256": header["scope_sha256"],
        "backend": header["backend"],
        "case": plan.case,
        "status": terminal["status"],
        "integrity_issues": sorted(set(issues)),
        "physical_checks_passed": [],
        "sender_timing": {
            "publications": len(pubs),
            "maximum_gap_ns": max(gaps) if gaps else None,
            "median_gap_ns": statistics.median(gaps) if gaps else None,
        },
        "controller_receipt_timing": {
            "available": False,
            "reason": "FR3_DIAG_V1 counters/peaks do not contain per-command receipt timestamps",
        },
        "frame_motion": frames,
        "controller_counter_deltas": counter_deltas,
        "controller_peaks": {
            key: max(d[key] for d in diagnostics) for key in DIAG_FIELDS[9:]
        }
        if diagnostics
        else {},
        "gripper_goals": goals,
        "gripper_state_samples": sum(r["source"] == "gripper_state" for r in telemetry),
        "gripper_status_samples": sum(
            r["source"] == "gripper_status"
            and (not goals or r["received_monotonic_ns"] >= goals[0]["monotonic_ns"])
            for r in telemetry
        ),
        "note": "Host receipt times align local streams; ROS source clocks remain separate. Counts and peaks do not certify hardware acceptance.",
    }
    return result


def reviewed_at(value: str) -> None:
    """Require an actual past UTC review timestamp."""
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if when.utcoffset().total_seconds() != 0 or when > datetime.now(timezone.utc):
            raise ValueError
    except (ValueError, AttributeError, TypeError) as exc:
        raise Blocked("Expected a past UTC operator review time") from exc


def validate_review(review_path: Path, session: Path) -> tuple[dict, dict]:
    """Bind a human verdict to exact session bytes and additional evidence."""
    review = json.loads(review_path.read_text())
    analysis = analyze_session(session)
    if analysis["backend"] != "ros":
        raise Blocked("Fake/offline evidence cannot approve physical mapping")
    if analysis["integrity_issues"] or analysis["status"] == "aborted":
        raise Blocked("Faulted or inconsistent session cannot pass a physical check")
    if (
        review.get("schema") != "fr3_mapping_operator_review_v1"
        or review.get("check") not in MAPPING_CHECKS
        or review.get("verdict") not in ("pass", "fail", "inconclusive")
    ):
        raise Blocked("Expected one named physical check and explicit operator verdict")
    for name in ("operator", "notes"):
        if not isinstance(review.get(name), str) or not review[name].strip():
            raise Blocked(f"Operator review requires {name}")
    reviewed_at(review.get("reviewed_at_utc"))
    ended = json.loads((session / "summary.json").read_text()).get("ended_at_utc")
    if ended and datetime.fromisoformat(
        review["reviewed_at_utc"].replace("Z", "+00:00")
    ) < datetime.fromisoformat(ended):
        raise Blocked("Operator review predates the captured calibration session")
    for name in ("session_manifest_sha256", "scope_sha256"):
        if review.get(name) != analysis[name]:
            raise Blocked(f"Review does not bind current {name}")
    for name in ("measurements", "criteria"):
        if not isinstance(review.get(name), dict) or not review[name]:
            raise Blocked(f"Operator review requires explicit {name}")
    if not review.get("attachments"):
        raise Blocked(
            "Attach raw measurement/geometry/acceptance/stop evidence with hashes"
        )
    for item in review["attachments"]:
        if checked_file(review_path.parent, item).stat().st_size == 0:
            raise Blocked("Operator evidence attachment must be nonempty")
    check = review["check"]
    if check in AXIS_CASES and (
        analysis["case"] != check or not analysis["frame_motion"]
    ):
        raise Blocked(
            "Signed-axis review requires that exact case and measured frame motion"
        )
    if check == "controller_acceptance" and (
        analysis["case"] not in AXIS_CASES or not analysis["controller_counter_deltas"]
    ):
        raise Blocked(
            "Acceptance review requires signed-axis motion and diagnostic deltas"
        )
    if check == "timing_10hz_100hz" and not analysis["sender_timing"]["maximum_gap_ns"]:
        raise Blocked(
            "Timing review requires publication intervals plus attached controller-local receipt evidence"
        )
    if check == "neutral_stop":
        events = read_rows(session / "events.jsonl")
        starts = [e for e in events if e["event"] == "case_started"]
        stops = [e for e in events if e["event"] == "operator_stop"]
        duration = json.loads((session / "session.json").read_text())["plan"][
            "duration_ns"
        ]
        if (
            not starts
            or not stops
            or not starts[0]["monotonic_ns"]
            < stops[0]["monotonic_ns"]
            < starts[0]["monotonic_ns"] + duration
        ):
            raise Blocked(
                "Neutral/stop review needs an operator stop during an active sample and attached hardware-stop evidence"
            )
    if check == "gripper_open_close":
        if (
            analysis["case"] not in ("gripper_open", "gripper_close")
            or not analysis["gripper_goals"]
            or not analysis["gripper_status_samples"]
        ):
            raise Blocked(
                "Gripper review requires measured/status evidence and the counterpart open/close session attachment"
            )
        item = review.get("counterpart_session_manifest")
        if not isinstance(item, dict):
            raise Blocked(
                "Gripper review requires a hashed counterpart_session_manifest"
            )
        counterpart = checked_file(review_path.parent, item).parent
        other = analyze_session(counterpart)
        if (
            other["backend"] != "ros"
            or other["status"] != "completed"
            or other["integrity_issues"]
            or other["case"]
            != (
                "gripper_close"
                if analysis["case"] == "gripper_open"
                else "gripper_open"
            )
            or other["scope_sha256"] != analysis["scope_sha256"]
            or len(other["gripper_goals"]) != 1
            or not other["gripper_status_samples"]
        ):
            raise Blocked(
                "Counterpart gripper session is incomplete, fake or from a different scope"
            )
        clocks = [
            json.loads((p / "session.json").read_text())["clock_id"]
            for p in (session, counterpart)
        ]
        if (
            clocks[0] != clocks[1]
            or abs(
                analysis["gripper_goals"][0]["monotonic_ns"]
                - other["gripper_goals"][0]["monotonic_ns"]
            )
            < 800_000_000
        ):
            raise Blocked(
                "Gripper goals require a shared host clock and at least 0.8 s separation"
            )
    return review, analysis


def apply_review(
    record_path: Path, session: Path, review_path: Path, output: Path, console: Console
) -> None:
    """Create a portable partial record after exact interactive review consent."""
    review, analysis = validate_review(review_path, session)
    record = json.loads(record_path.read_text())
    record_digest = sha256(record_path)
    if set(record["checks"]) != set(MAPPING_CHECKS):
        raise Blocked("Expected the complete 17-check pending record")
    if (
        record.get("calibration_scope_sha256", analysis["scope_sha256"])
        != analysis["scope_sha256"]
    ):
        raise Blocked("Cannot combine different controller/configuration/tool scopes")
    if output.exists() or output.with_suffix(".evidence").exists():
        raise Blocked("Review output/evidence must be a new path")
    console.confirm(f"REVIEW {review['check']} {sha256(review_path)[:12]}")
    # Revalidate bytes after a potentially long console prompt.
    fresh, analysis = validate_review(review_path, session)
    if fresh != review or sha256(record_path) != record_digest:
        raise Blocked("Operator review or pending record changed during confirmation")
    bundle = output.with_suffix(".evidence")
    bundle.mkdir()
    shutil.copytree(session, bundle / "session")
    verify_session(bundle / "session")
    shutil.copyfile(review_path, bundle / "operator_review.source.json")
    attachments = []
    for item in review["attachments"]:
        source = checked_file(review_path.parent, item)
        target = bundle / item["sha256"]
        if not target.exists():
            shutil.copyfile(source, target)
        if sha256(target) != item["sha256"]:
            raise Blocked("Review attachment changed while copying")
        attachments.append(file_record(target, bundle))
    normalized = {**review, "attachments": attachments}
    if "counterpart_session_manifest" in review:
        counterpart = checked_file(
            review_path.parent, review["counterpart_session_manifest"]
        ).parent
        shutil.copytree(counterpart, bundle / "counterpart")
        verify_session(bundle / "counterpart")
        normalized["counterpart_session_manifest"] = file_record(
            bundle / "counterpart/manifest.json", bundle
        )
    write_json(bundle / "operator_review.json", normalized)
    proof = {
        "schema": "fr3_mapping_review_evidence_v1",
        "operator_review": file_record(bundle / "operator_review.json", bundle),
        "operator_review_source": file_record(
            bundle / "operator_review.source.json", bundle
        ),
        "session_manifest": file_record(bundle / "session/manifest.json", bundle),
        "attachments": attachments,
        "analysis": analysis,
    }
    write_json(bundle / "review_evidence.json", proof)
    record["calibration_scope_sha256"] = analysis["scope_sha256"]
    scope = json.loads((session / "session.json").read_text())["scope"]
    record.update(
        {
            key: scope[key]
            for key in (
                "controller_binary_sha256",
                "controller_config_sha256",
                "base_frame",
                "tool_frame",
            )
        }
    )
    record["operator"] = None
    record["verified_at_utc"] = None
    record["physical_execution_approved"] = False
    record["status"] = "pending_final_operator_approval"
    # Keep earlier relative evidence links valid when a partial record moves.
    for check in record["checks"].values():
        if check.get("evidence_path"):
            check["evidence_path"] = os.path.relpath(
                (record_path.parent / check["evidence_path"]).resolve(), output.parent
            )
    record["checks"][review["check"]] = {
        "passed": review["verdict"] == "pass",
        "status": review["verdict"],
        "evidence_path": os.path.relpath(
            bundle / "review_evidence.json", output.parent
        ),
        "evidence_sha256": sha256(bundle / "review_evidence.json"),
    }
    write_json(output, record)


def audit_record(path: Path) -> dict:
    """Check every existing proof transitively and report pending physical checks."""
    record = json.loads(path.read_text())
    if (
        record.get("schema") != "fr3_pi05_mapping_verification_v1"
        or record.get("mapping_mode") != "identity_verified"
        or record.get("arm_axes") != list(ARM_AXES)
        or record.get("policy_period_ns") != PERIOD_NS
        or record.get("publish_period_ns") != 10_000_000
        or set(record.get("checks", {})) != set(MAPPING_CHECKS)
    ):
        raise Blocked("Expected the complete six-axis 10 Hz/100 Hz mapping record")
    pending = []
    for key in MAPPING_CHECKS:
        check = record["checks"][key]
        if check.get("passed") is not True:
            pending.append(key)
            if not check.get("evidence_path"):
                continue
        proof_path = checked_file(
            path.parent,
            {"path": check["evidence_path"], "sha256": check["evidence_sha256"]},
        )
        proof = json.loads(proof_path.read_text())
        if proof.get("schema") != "fr3_mapping_review_evidence_v1":
            raise Blocked("Expected a reviewed calibration evidence bundle")
        review_path = checked_file(
            proof_path.parent, proof["operator_review"], confined=True
        )
        source_path = checked_file(
            proof_path.parent, proof["operator_review_source"], confined=True
        )
        original = json.loads(source_path.read_text())
        normalized = json.loads(review_path.read_text())
        for k in set(original) | set(normalized):
            if k not in (
                "attachments",
                "counterpart_session_manifest",
            ) and original.get(k) != normalized.get(k):
                raise Blocked(
                    "Portable review differs from the original operator verdict"
                )
        if [x["sha256"] for x in original["attachments"]] != [
            x["sha256"] for x in normalized["attachments"]
        ]:
            raise Blocked(
                "Portable review attachments differ from the operator's evidence"
            )
        if original.get("counterpart_session_manifest", {}).get(
            "sha256"
        ) != normalized.get("counterpart_session_manifest", {}).get("sha256"):
            raise Blocked("Portable counterpart differs from the operator review")
        manifest = checked_file(
            proof_path.parent, proof["session_manifest"], confined=True
        )
        review, analysis = validate_review(review_path, manifest.parent)
        if (
            review["check"] != key
            or (review["verdict"] == "pass") != (check.get("passed") is True)
            or analysis["scope_sha256"] != record["calibration_scope_sha256"]
        ):
            raise Blocked(
                "Passed check differs from its operator review or installation scope"
            )
        scope = verify_session(manifest.parent)["header"]["scope"]
        if any(
            record.get(k) != scope[k]
            for k in (
                "controller_binary_sha256",
                "controller_config_sha256",
                "base_frame",
                "tool_frame",
            )
        ):
            raise Blocked(
                "Mapping record installation/frame fields differ from physical evidence"
            )
        for item in proof["attachments"]:
            checked_file(proof_path.parent, item, confined=True)
    return {
        "required": len(MAPPING_CHECKS),
        "passed": len(MAPPING_CHECKS) - len(pending),
        "pending": pending,
        "physical_execution_approved": record.get("physical_execution_approved")
        is True,
    }


def finalize_record(path: Path, output: Path, operator: str, console: Console) -> None:
    """Approve a complete record only after all physical proofs and final consent."""
    if not operator.strip() or audit_record(path)["pending"]:
        raise Blocked(
            "Final approval requires a named operator and all 17 physical reviews"
        )
    before = sha256(path)
    console.confirm(f"APPROVE MAPPING {before[:12]}")
    if sha256(path) != before or audit_record(path)["pending"]:
        raise Blocked("Mapping record changed during final approval")
    record = json.loads(path.read_text())
    for check in record["checks"].values():
        check["evidence_path"] = os.path.relpath(
            (path.parent / check["evidence_path"]).resolve(), output.parent
        )
    record.update(
        operator=operator,
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
        physical_execution_approved=True,
        status="operator_approved",
    )
    write_json(output, record)
    load_mapping_verification(output)


def main() -> None:
    """Expose non-actuating capture verification and explicit review commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("config-manifest")
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("files", type=Path, nargs="+")
    for name in ("verify-session", "analyze"):
        p = commands.add_parser(name)
        p.add_argument("session", type=Path)
    for name in ("audit-record", "verify-record"):
        p = commands.add_parser(name)
        p.add_argument("record", type=Path)
    apply = commands.add_parser("apply-review")
    apply.add_argument("--record", type=Path, required=True)
    apply.add_argument("--session", type=Path, required=True)
    apply.add_argument("--review", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    final = commands.add_parser("finalize")
    final.add_argument("--record", type=Path, required=True)
    final.add_argument("--operator", required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "config-manifest":
        if any(not p.is_file() or p.stat().st_size == 0 for p in args.files):
            parser.error("Configuration manifest inputs must be nonempty files")
        write_json(
            args.output,
            {
                "schema": "fr3_mapping_config_manifest_v1",
                "files": [
                    file_record(p.resolve(), args.output.parent.resolve())
                    for p in args.files
                ],
            },
        )
        result = {"sha256": sha256(args.output)}
    elif args.command == "analyze":
        result = analyze_session(args.session)
    elif args.command == "verify-session":
        verified = verify_session(args.session)
        result = {
            "manifest_sha256": verified["manifest_sha256"],
            "backend": verified["header"]["backend"],
            "physical_checks_passed": [],
        }
    elif args.command in ("audit-record", "verify-record"):
        result = audit_record(args.record)
        if args.command == "verify-record":
            result["record_sha256"] = load_mapping_verification(args.record)[
                "record_sha256"
            ]
    elif args.command == "apply-review":
        apply_review(args.record, args.session, args.review, args.output, Console())
        result = audit_record(args.output)
    else:
        finalize_record(args.record, args.output, args.operator, Console())
        result = audit_record(args.output)
    sys.stdout.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
