# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Calibrate through fake ROS endpoints; no controller or hardware access."""

import io
import json
import math
import os
import pty
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from robot_mocks.mapping_calibration import FakeConsole, FakeRuntime, deployment_file

from toolkits.realworld_check.fr3_mapping_calibration import (
    AXIS_CASES,
    DIAG_FIELDS,
    Blocked,
    Console,
    Journal,
    Plan,
    _local_claim,
    check_endpoints,
    compose,
    execute_case,
    load_deployment,
    pose,
    rotation_vector,
)
from toolkits.realworld_check.fr3_mapping_evidence import (
    analyze_session,
    verify_session,
)


@pytest.fixture
def setup(tmp_path):
    path = deployment_file(tmp_path / "deployment")
    return path, load_deployment(path)


@pytest.mark.parametrize("case", AXIS_CASES)
def test_all_signed_axes_use_distinct_100ms_slots_at_gain_one(setup, tmp_path, case):
    path, deployment = setup
    runtime = FakeRuntime(deployment)
    plan = Plan(case)
    output = tmp_path / "case"
    summary = execute_case(runtime, plan, path, output, FakeConsole(runtime))
    assert summary["status"] == "completed"
    assert summary["commands"] == 5
    assert summary["gripper_goals"] == 0
    assert summary["cleanup_neutral_publications"] == 5
    assert runtime.gripper_messages == []
    nonzero = [(t, a) for t, a in runtime.arm_messages if any(a)]
    begin = next(
        json.loads(line)["monotonic_ns"]
        for line in (output / "events.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "case_started"
    )
    axis, direction = case.rsplit("_", 1)
    column = (
        "linear_x",
        "linear_y",
        "linear_z",
        "angular_x",
        "angular_y",
        "angular_z",
    ).index(axis)
    for t, value in nonzero:
        index = (t - begin) // 100_000_000
        expected = (
            0.03 * (1, 0.8, 0.6, 0.8, 1)[index] * (1 if direction == "positive" else -1)
        )
        assert value[column] == expected
        assert all(x == 0 for i, x in enumerate(value) if i != column)
    assert all(not any(a) for t, a in runtime.arm_messages if t >= begin + 500_000_000)
    report = analyze_session(output)
    assert report["integrity_issues"] == []
    assert report["sender_timing"]["median_gap_ns"] == 10_000_000
    assert report["sender_timing"]["maximum_gap_ns"] <= 50_000_000
    assert report["physical_checks_passed"] == []
    assert not report["controller_receipt_timing"]["available"]
    assert verify_session(output)["header"]["backend"] == "fake"


@pytest.mark.parametrize(
    "case,value",
    [("gripper_open", 1.0), ("gripper_close", 0.0), ("gripper_hold", None)],
)
def test_gripper_case_has_only_one_filtered_goal_and_neutral_arm(
    setup, tmp_path, case, value
):
    path, deployment = setup
    runtime = FakeRuntime(deployment, gripper=True)
    result = execute_case(
        runtime, Plan(case), path, tmp_path / "case", FakeConsole(runtime)
    )
    assert result["status"] == "completed"
    assert all(not any(a) for _, a in runtime.arm_messages)
    assert [v for _, v in runtime.gripper_messages] == (
        [] if value is None else [value]
    )
    report = analyze_session(tmp_path / "case")
    assert not report["integrity_issues"]
    if value is not None:
        assert runtime.gripper_messages[0][0] - runtime.enabled_at >= 200_000_000


def test_neutral_case_never_creates_a_gripper_goal_or_arm_motion(setup, tmp_path):
    path, deployment = setup
    runtime = FakeRuntime(deployment)
    summary = execute_case(
        runtime, Plan("neutral"), path, tmp_path / "case", FakeConsole(runtime)
    )
    assert summary["status"] == "completed"
    assert runtime.arm_messages and all(not any(a) for _, a in runtime.arm_messages)
    assert not runtime.gripper_messages


def test_operator_stop_is_processed_before_next_publication(setup, tmp_path):
    path, deployment = setup
    runtime = FakeRuntime(deployment)
    console = FakeConsole(runtime, stop_after_ns=350_000_000)
    summary = execute_case(
        runtime, Plan("linear_x_positive"), path, tmp_path / "case", console
    )
    assert summary["status"] == "operator_stop"
    assert any(any(a) for _, a in runtime.arm_messages)
    assert all(
        not any(a)
        for t, a in runtime.arm_messages
        if t >= runtime.enabled_at + 350_000_000
    )
    report = analyze_session(tmp_path / "case")
    assert report["integrity_issues"] == []


@pytest.mark.parametrize("when", [1, 190_000_000])
def test_stop_prevents_a_gripper_goal_before_third_confirmation(setup, tmp_path, when):
    path, deployment = setup
    runtime = FakeRuntime(deployment, gripper=True)
    execute_case(
        runtime,
        Plan("gripper_close"),
        path,
        tmp_path / "case",
        FakeConsole(runtime, stop_after_ns=when),
    )
    assert not runtime.gripper_messages


def test_cancelled_consent_never_creates_command_publishers(setup, tmp_path):
    path, deployment = setup
    runtime = FakeRuntime(deployment)
    with pytest.raises(Blocked, match="canceled"):
        execute_case(
            runtime,
            Plan("linear_x_positive"),
            path,
            tmp_path / "case",
            FakeConsole(runtime, approve=False),
        )
    assert runtime.arm_pub is None and not (tmp_path / "case").exists()


def test_long_operator_prompt_requires_a_new_full_baseline(setup, tmp_path):
    path, deployment = setup
    runtime = FakeRuntime(deployment)
    console = FakeConsole(runtime, confirmation_delay_ns=7_000_000_000)
    execute_case(runtime, Plan("linear_x_positive"), path, tmp_path / "case", console)
    header = verify_session(tmp_path / "case")["header"]
    assert runtime.arm_messages[0][0] - header["started_monotonic_ns"] >= 2_100_000_000


@pytest.mark.parametrize(
    "fault",
    [
        "competing",
        "extra_subscriber",
        "missing_subscriber",
        "stall",
        "rotation",
        "translation",
        "pose_frame",
        "fault",
        "inactive",
        "claim",
        "geometry",
        "nonfinite",
        "counter_reset",
    ],
)
def test_live_fault_revokes_motion_and_preserves_neutral_cleanup(
    setup, tmp_path, fault
):
    path, deployment = setup
    runtime = FakeRuntime(deployment)
    tripped = False

    def hook(r):
        nonlocal tripped
        if tripped or r.enabled_at is None or r.now_ns() - r.enabled_at < 50_000_000:
            return
        tripped = True
        if fault in ("competing", "extra_subscriber", "missing_subscriber"):
            setattr(r, fault, True)
        elif fault == "stall":
            r.clock += 60_000_000
        elif fault == "rotation":
            r.measured_pose[3:] = [math.sin(0.02 / 2), 0, 0, math.cos(0.02 / 2)]
        elif fault == "translation":
            r.measured_pose[0] += 0.02
        elif fault == "counter_reset":
            r.diag = dict.fromkeys(DIAG_FIELDS, -1)
            r.last_diag = 0
        else:

            def mutate(m):
                if fault == "pose_frame":
                    m["pose"]["header"]["frame_id"] = "wrong_frame"
                elif fault == "fault":
                    m["robot_state"]["current_errors"]["joint_reflex"] = True
                elif fault == "inactive":
                    m["controller"]["state"] = "inactive"
                elif fault == "claim":
                    m["controller"]["claimed_interfaces"].pop()
                elif fault == "geometry":
                    m["robot_state"]["f_t_ee"]["pose"]["position"]["x"] = 0.1
                else:
                    m["pose"]["pose"]["position"]["x"] = float("nan")

            r.mutate = mutate

    runtime.hook = hook
    output = tmp_path / "case"
    summary = execute_case(
        runtime, Plan("linear_x_positive"), path, output, FakeConsole(runtime)
    )
    assert tripped and summary["status"] == "aborted"
    assert all(not any(a) for _, a in runtime.arm_messages[-5:])
    assert (output / "summary.json").is_file()
    verify_session(output)


@pytest.mark.parametrize(
    "source",
    [
        "pose",
        "measured_joints",
        "desired_joints",
        "robot_state",
        "controller",
        "diagnostics",
        "gripper_state",
    ],
)
def test_every_required_source_has_a_live_freshness_guard(setup, tmp_path, source):
    path, deployment = setup
    runtime = FakeRuntime(deployment, gripper=True)

    def hook(r):
        if r.enabled_at is not None:
            r.omit.add(source)
            if source in r.telemetry.latest:
                r.telemetry.latest[source]["received_monotonic_ns"] = 1

    runtime.hook = hook
    summary = execute_case(
        runtime, Plan("gripper_close"), path, tmp_path / "case", FakeConsole(runtime)
    )
    assert summary["status"] == "aborted"
    assert not runtime.gripper_messages
    assert all(not any(a) for _, a in runtime.arm_messages)


@pytest.mark.parametrize("stop_during_construction", [False, True])
def test_late_message_construction_cannot_publish_expired_or_stopped_motion(
    setup, tmp_path, stop_during_construction
):
    path, deployment = setup
    runtime = FakeRuntime(deployment)
    console = FakeConsole(
        runtime, stop_after_ns=20_000_000 if stop_during_construction else None
    )
    runtime.hook = lambda r: (
        setattr(r, "construct_delay_ns", 110_000_000)
        if r.enabled_at is not None
        else None
    )
    summary = execute_case(
        runtime, Plan("linear_x_positive"), path, tmp_path / "case", console
    )
    assert summary["status"] == "aborted"
    assert runtime.arm_messages and all(not any(a) for _, a in runtime.arm_messages)


@pytest.mark.parametrize(
    "magnitude,duration",
    [
        (0, 500_000_000),
        (0.050001, 500_000_000),
        (-0.03, 500_000_000),
        (float("nan"), 500_000_000),
        (0.03, 600_000_000),
        (0.03, 150_000_000),
        (0.03, 0),
    ],
)
def test_invalid_case_bounds_never_reach_hardware(magnitude, duration):
    with pytest.raises(Blocked):
        Plan("linear_x_positive", magnitude, duration)


@pytest.mark.parametrize(
    "field",
    [
        "operator",
        "hardware_stop_checked",
        "workspace_clear",
        "live_installation_matched",
        "frames_calibrated",
        "historical_errors_reviewed",
        "rotation_limit_rad",
        "translation_limit_m",
    ],
)
def test_deployment_must_be_explicit_and_complete(setup, field):
    path, _ = setup
    record = json.loads(path.read_text())
    record[field] = None
    path.write_text(json.dumps(record))
    with pytest.raises((Blocked, TypeError)):
        load_deployment(path)


def test_changed_copied_binary_is_not_accepted(setup):
    path, _ = setup
    (path.parent / "FAKE_CONTROLLER.so").write_bytes(b"changed")
    with pytest.raises(Blocked, match="hash mismatch"):
        load_deployment(path)


def test_consent_change_race_never_creates_publishers(setup, tmp_path):
    path, deployment = setup
    runtime = FakeRuntime(deployment)
    console = FakeConsole(runtime)

    def confirm(token):
        record = json.loads(path.read_text())
        record["operator"] = "another operator"
        path.write_text(json.dumps(record))

    console.confirm = confirm
    with pytest.raises(Blocked, match="changed"):
        execute_case(
            runtime, Plan("linear_x_positive"), path, tmp_path / "case", console
        )
    assert not runtime.arm_messages and runtime.arm_pub is None


def test_interactive_tty_is_required_even_with_exact_piped_token(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("CALIBRATE linear_x_positive abc\n"))
    with pytest.raises(Blocked, match="TTY"):
        Console().confirm("CALIBRATE linear_x_positive abc")


@pytest.mark.parametrize("correct", [True, False])
def test_real_terminal_confirmation_requires_the_exact_current_token(
    monkeypatch, correct
):
    master, slave = pty.openpty()
    terminal = os.fdopen(slave, "r")
    try:
        monkeypatch.setattr(sys, "stdin", terminal)
        token = "CALIBRATE linear_x_positive CURRENT_TEST_TOKEN"
        os.write(master, ((token if correct else "old token") + "\n").encode())
        if correct:
            Console().confirm(token)
        else:
            with pytest.raises(Blocked, match="did not approve"):
                Console().confirm(token)
    finally:
        terminal.close()
        os.close(master)


@pytest.mark.parametrize("line", [b"\n", b""])
def test_enter_or_terminal_eof_revokes_command_authority(monkeypatch, line):
    reader, writer = os.pipe()
    stream = os.fdopen(reader, "r")
    try:
        monkeypatch.setattr(sys, "stdin", stream)
        os.write(writer, line)
        os.close(writer)
        writer = None
        console = Console()
        assert console.stopped()
        assert console.stopped()
    finally:
        stream.close()
        if writer is not None:
            os.close(writer)


def test_endpoint_identity_type_and_exclusivity():
    def peer(name="_NODE_NAME_UNKNOWN_", cls="geometry_msgs/msg/Twist"):
        return SimpleNamespace(node_name=name, topic_type=cls)

    check_endpoints(
        [],
        [peer()],
        own=False,
        topic_type="geometry_msgs/msg/Twist",
        peer="joint_impedance_ik_controller",
    )
    for writers, readers in (
        ([peer("spacemouse")], [peer()]),
        ([], [peer(cls="std_msgs/msg/Float32")]),
        ([], [peer("other_controller")]),
        ([], [peer(), peer()]),
    ):
        with pytest.raises(Blocked):
            check_endpoints(
                writers,
                readers,
                own=False,
                topic_type="geometry_msgs/msg/Twist",
                peer="joint_impedance_ik_controller",
            )


def test_geometry_recovers_local_rotation_and_tcp_lever_arm():
    a = pose([0, 0, 0, 0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    local = pose([0, 0, 0, math.sin(0.001 / 2), 0, 0, math.cos(0.001 / 2)])
    b = compose(a, local)
    assert rotation_vector(a, b) == pytest.approx((0.001, 0, 0), abs=1e-12)
    assert rotation_vector(
        a, tuple(-x if i >= 3 else x for i, x in enumerate(a))
    ) == pytest.approx((0, 0, 0), abs=1e-12)
    tcp = compose(b, pose([0, 0, 0.2, 0, 0, 0, 1]))
    assert tcp[0] > 0 and tcp[2] < 0.2


def test_default_cli_prints_only_plan_without_importing_ros():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "toolkits.realworld_check.fr3_mapping_calibration"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["physical_execution_approved"] is False
    assert payload["plan"]["case"] == "linear_x_positive"


def test_local_claim_is_exclusive_and_released(tmp_path):
    path = tmp_path / "claim.lock"
    with _local_claim(path):
        with pytest.raises(Blocked, match="Another local calibration"):
            with _local_claim(path):
                pytest.fail("Second process acquired the live calibration claim")
    with _local_claim(path):
        pass


def test_evidence_queue_is_bounded_and_closes_after_overflow(tmp_path):
    journal = Journal(tmp_path / "capture", capacity=1)
    with pytest.raises(Blocked, match="queue overflow"):
        for _ in range(10000):
            journal.add("events", {"payload": "x" * 1_000_000})
    journal.close()


def test_storage_failure_stops_commands_but_does_not_block_neutral_cleanup(
    setup, tmp_path
):
    path, deployment = setup
    runtime = FakeRuntime(deployment)
    closed = False

    def hook(r):
        nonlocal closed
        if (
            not closed
            and r.enabled_at is not None
            and r.now_ns() - r.enabled_at >= 50_000_000
        ):
            closed = True
            r.journal.streams["events"].close()

    runtime.hook = hook
    summary = execute_case(
        runtime,
        Plan("linear_x_positive"),
        path,
        tmp_path / "case",
        FakeConsole(runtime),
    )
    assert closed and summary["status"] == "aborted"
    assert "capture_error" in summary
    assert all(not any(a) for _, a in runtime.arm_messages[-5:])


def test_peg_lever_arm_translation_guard_applies_during_small_rotation(setup, tmp_path):
    path, _ = setup
    raw = json.loads(path.read_text())
    raw["translation_limit_m"] = 0.0001
    path.write_text(json.dumps(raw))
    runtime = FakeRuntime(load_deployment(path))

    def hook(r):
        if r.enabled_at is not None and r.now_ns() - r.enabled_at >= 50_000_000:
            r.measured_pose[3:] = [math.sin(0.003 / 2), 0, 0, math.cos(0.003 / 2)]

    runtime.hook = hook
    result = execute_case(
        runtime,
        Plan("angular_x_positive"),
        path,
        tmp_path / "case",
        FakeConsole(runtime),
    )
    assert result["status"] == "aborted"
    assert "tcp translation" in result["reason"]


def test_action_status_is_not_treated_as_a_state_heartbeat(setup, tmp_path):
    path, deployment = setup
    runtime = FakeRuntime(deployment, gripper=True)
    runtime.omit.add("gripper_status")
    result = execute_case(
        runtime, Plan("gripper_open"), path, tmp_path / "case", FakeConsole(runtime)
    )
    assert result["status"] == "completed"
    assert analyze_session(tmp_path / "case")["gripper_status_samples"] == 0


@pytest.mark.parametrize("duration", [100_000_000, 200_000_000])
def test_gripper_goal_case_requires_three_policy_samples(duration):
    with pytest.raises(Blocked, match="three policy samples"):
        Plan("gripper_open", duration_ns=duration)


def test_module_cli_preflight_handles_package_exception_identity(
    setup, monkeypatch, capsys
):
    from toolkits.realworld_check import fr3_mapping_calibration_ros

    path, _ = setup
    instances = []

    class Context(FakeRuntime):
        def __enter__(self):
            instances.append(self)
            return self

        def __exit__(self, *_args):
            pass

        def check_graph(self, *, own):
            self.missing_subscriber = self.spins < 100
            return super().check_graph(own=own)

    monkeypatch.setattr(fr3_mapping_calibration_ros, "RosRuntime", Context)
    monkeypatch.setenv("ROS_DOMAIN_ID", "70")
    monkeypatch.setattr(
        sys, "argv", ["calibration", "--deployment", str(path), "--preflight"]
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module(
            "toolkits.realworld_check.fr3_mapping_calibration", run_name="__main__"
        )
    assert json.loads(capsys.readouterr().out)["robot_commands_sent"] == 0
    assert instances[0].arm_pub is None


def test_ros_context_cleanup_continues_after_an_endpoint_failure(setup):
    from toolkits.realworld_check.fr3_mapping_calibration_ros import RosRuntime

    _, deployment = setup
    runtime = RosRuntime(deployment, gripper=False)
    closed = []

    def failed(**_kwargs):
        closed.append("executor")
        raise RuntimeError("fake ROS shutdown failure")

    runtime.executors = [SimpleNamespace(shutdown=failed)]
    runtime.nodes = [SimpleNamespace(destroy_node=lambda: closed.append("node"))]
    runtime.contexts = [SimpleNamespace(try_shutdown=lambda: closed.append("context"))]
    with pytest.raises(Blocked, match="cleanup failed"):
        runtime.__exit__()
    assert closed == ["executor", "node", "context"]
    runtime.__exit__()
