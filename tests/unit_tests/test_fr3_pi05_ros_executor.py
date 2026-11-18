# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Offline request-boundary tests. No ROS or robot access."""

import time

import pytest

from toolkits.realworld_check.fr3_pi05_ros_executor import validate_chunk


def test_accepts_valid_policy_chunk():
    now = time.monotonic_ns()
    chunk = [[0.5, -0.5, 0, 0, 0, 0, 1.04] for _ in range(10)]
    assert (
        len(
            validate_chunk(
                chunk,
                observation_ns=now - 100_000_000,
                issued_ns=now - 10_000_000,
                now_ns=now,
            )
        )
        == 10
    )


@pytest.mark.parametrize(
    "bad",
    [
        [[0.0] * 7] * 9,
        [[1.01, 0, 0, 0, 0, 0, 0]] * 10,
        [[0, 0, 0, 0, 0, 0, 1.5]] * 10,
        [[float("nan"), 0, 0, 0, 0, 0, 0]] * 10,
    ],
)
def test_rejects_bad_requests(bad):
    now = time.monotonic_ns()
    with pytest.raises(ValueError):
        validate_chunk(
            bad, observation_ns=now - 10_000_000, issued_ns=now - 1_000_000, now_ns=now
        )


def test_rejects_old_observation():
    now = time.monotonic_ns()
    with pytest.raises(ValueError, match="Stale"):
        validate_chunk(
            [[0.0] * 7] * 10,
            observation_ns=now - 2_000_000_000,
            issued_ns=now - 1_000_000,
            now_ns=now,
        )


@pytest.mark.parametrize("delay_ms", [300, 484, 1500])
def test_schedule_expires_to_neutral_across_inference_delay(delay_ms):
    from toolkits.realworld_check.fr3_pi05_ros_executor import CommandSchedule

    schedule = CommandSchedule()
    start = 2_000_000_000
    values = [[i / 10, 0, 0, 0, 0, 0, 1] for i in range(10)]
    schedule.start(values, observation_ns=start, issued_ns=start, now_ns=start)
    indices = []
    for offset in range(0, 1000 + delay_ms, 10):
        sample = schedule.sample(start + offset * 1_000_000)
        indices.append(None if sample is None else sample[0])
    assert indices[:100] == [i for i in range(10) for _ in range(10)]
    assert all(i is None for i in indices[100:])
    assert not schedule.batch
    later = start + (1000 + delay_ms) * 1_000_000
    with pytest.raises(ValueError, match="Stale"):
        schedule.start(values, observation_ns=start, issued_ns=later, now_ns=later)
    schedule.start(values, observation_ns=later, issued_ns=later, now_ns=later)
    assert schedule.sample(later) == (0, values[0])


def test_schedule_stall_revokes_chunk_without_catching_up():
    from toolkits.realworld_check.fr3_pi05_ros_executor import CommandSchedule

    schedule = CommandSchedule()
    start = 2_000_000_000
    schedule.start(
        [[0.5] * 7] * 10, observation_ns=start, issued_ns=start, now_ns=start
    )
    assert schedule.sample(start)[0] == 0
    with pytest.raises(RuntimeError, match="deadline"):
        schedule.sample(start + 60_000_000)
    assert schedule.sample(start + 61_000_000) is None


@pytest.fixture
def ros_runtime(monkeypatch, capsys):
    """Replace only the ROS process edge; run the real executor loop."""
    import json
    import os
    import sys
    from types import ModuleType, SimpleNamespace

    from toolkits.realworld_check import fr3_pi05_ros_executor as executor

    class Twist:
        def __init__(self):
            runtime.ns += runtime.message_delay_ns
            runtime.message_delay_ns = 0
            self.linear = SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.angular = SimpleNamespace(x=0.0, y=0.0, z=0.0)

    class Runtime:
        ns = 2_000_000_000
        steps = 0
        writes = []
        pose_callback = None
        pose_x = 0.0
        message_delay_ns = 0
        stale_at = None
        failure_at = None
        failure = None
        stall_at = None
        pending = []
        publications = []

        def __init__(self):
            read_fd, self.write_fd = os.pipe()
            self.stdin = os.fdopen(read_fd, "r")

        def send(self, op, **fields):
            os.write(self.write_fd, (json.dumps({"op": op, **fields}) + "\n").encode())

        def chunk(self):
            self.send(
                "chunk",
                values=[[0.4, 0, 0, 0, 0, 0, 0]] * 10,
                observation_ns=self.ns,
                issued_ns=self.ns,
            )

        def spin(self, node, timeout_sec):
            self.ns += 2_000_000
            self.steps += 1
            if self.stall_at is not None and self.ns >= self.stall_at:
                self.ns += 100_000_000
                self.stall_at = None
            if self.pose_callback and (
                self.stale_at is None or self.ns < self.stale_at
            ):
                self.pose_callback(
                    SimpleNamespace(
                        pose=SimpleNamespace(
                            position=SimpleNamespace(x=self.pose_x, y=0.0, z=0.0)
                        )
                    )
                )
            for at, callback in list(self.pending):
                if self.ns >= at:
                    self.pending.remove((at, callback))
                    callback()

        def create_subscription(self, cls, topic, callback, qos):
            self.pose_callback = callback

        def create_publisher(self, cls, topic, depth):
            self.writes.append((topic, depth))

            def publish(msg):
                values = (
                    (
                        msg.linear.x,
                        msg.linear.y,
                        msg.linear.z,
                        msg.angular.x,
                        msg.angular.y,
                        msg.angular.z,
                    )
                    if hasattr(msg, "linear")
                    else msg.data
                )
                if (
                    self.failure == "publish"
                    and self.ns >= self.failure_at
                    and topic == executor.ARM_TOPIC
                    and any(values)
                ):
                    raise RuntimeError("Simulated arm publisher failure")
                self.publications.append((self.ns, topic, values))

            return SimpleNamespace(publish=publish)

        def get_publishers_info_by_topic(self, topic):
            if self.failure == "exclusive" and self.ns >= self.failure_at:
                return [SimpleNamespace(node_name="spacemouse")]
            return [SimpleNamespace(node_name=executor.NAME)] if self.writes else []

        def get_subscriptions_info_by_topic(self, topic):
            if self.failure == "subscriber" and self.ns >= self.failure_at:
                return []
            return [object()]

        def events(self):
            return [json.loads(line) for line in capsys.readouterr().out.splitlines()]

        def destroy_node(self):
            pass

    runtime = Runtime()
    modules = {
        "rclpy": {
            "init": lambda: None,
            "shutdown": lambda: None,
            "spin_once": runtime.spin,
        },
        "rclpy.node": {"Node": lambda *a, **kw: runtime},
        "rclpy.qos": {"qos_profile_sensor_data": object()},
        "geometry_msgs.msg": {"PoseStamped": object, "Twist": Twist},
        "std_msgs.msg": {"Float32": lambda: SimpleNamespace(data=0.0)},
    }
    for name, members in modules.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("ROS_DOMAIN_ID", "70")
    monkeypatch.delenv("FR3_PI05_MAPPING_MODE", raising=False)
    monkeypatch.setattr(sys, "stdin", runtime.stdin)
    monkeypatch.setattr(
        executor,
        "time",
        SimpleNamespace(
            monotonic=lambda: runtime.ns / 1e9, monotonic_ns=lambda: runtime.ns
        ),
    )
    # Discovery consumes 13 spins; input arrives only after ready.
    runtime.pending = [
        (2_030_000_000, lambda: runtime.send("begin_episode")),
        (2_032_000_000, runtime.chunk),
        (4_000_000_000, lambda: runtime.send("quit")),
    ]
    try:
        yield runtime, executor
    finally:
        runtime.stdin.close()
        os.close(runtime.write_fd)


def test_real_loop_refreshes_100hz_and_neutralizes_chunk_boundary(ros_runtime):
    runtime, executor = ros_runtime
    assert executor.main() == 0
    events = runtime.events()
    actions = [e for e in events if e["event"] == "action"]
    assert [e["command_index"] for e in actions] == list(range(10))
    assert all(e["sent_arm_6d"][0] == pytest.approx(0.06) for e in actions)
    assert [e["gripper_event"] for e in actions[:3]] == ["hold", "hold", "close"]
    arm = [
        (t, v) for t, topic, v in runtime.publications if topic == executor.ARM_TOPIC
    ]
    active = [(t, v) for t, v in arm if 2_032_000_000 <= t < 4_000_000_000]
    assert max(b[0] - a[0] for a, b in zip(active, active[1:])) <= 10_000_000
    assert len([v for t, v in active if any(v)]) == 100
    assert all(not any(v) for t, v in active if t >= 3_032_000_000)
    assert (executor.ARM_TOPIC, 1) in runtime.writes
    assert len([e for e in events if e["event"] == "chunk_done"]) == 1


@pytest.mark.parametrize(
    "failure",
    [
        "pose",
        "exclusive",
        "subscriber",
        "publish",
        "stall",
        "construction",
        "operator",
        "invalid",
        "partial",
        "eof",
    ],
)
def test_real_loop_never_resumes_motion_after_stop_or_fault(ros_runtime, failure):
    import os

    runtime, executor = ros_runtime
    at = 2_400_000_000
    if failure == "pose":
        runtime.stale_at = at
        cutoff = at + 250_000_000
    elif failure in ("exclusive", "subscriber", "publish"):
        runtime.failure, runtime.failure_at = failure, at
        cutoff = at
    elif failure == "stall":
        runtime.stall_at = at
        cutoff = at
    elif failure == "construction":
        runtime.pending.append(
            (
                at,
                lambda: setattr(runtime, "message_delay_ns", 100_000_000),
            )
        )
        cutoff = at
    elif failure == "operator":
        runtime.pending.append((at, lambda: runtime.send("end_episode")))
        cutoff = at
    elif failure == "invalid":
        runtime.pending.append((at, lambda: runtime.send("unknown_operation")))
        cutoff = at
    elif failure == "partial":
        runtime.pending.append((at, lambda: os.write(runtime.write_fd, b'{"op":')))
        cutoff = 3_032_000_000
    else:
        # Closing the writer represents parent death, including mid-chunk.
        runtime.pending = [(t, cb) for t, cb in runtime.pending if t < at]

        def disconnect():
            os.close(runtime.write_fd)
            runtime.write_fd = os.open(os.devnull, os.O_WRONLY)

        runtime.pending.append((at, disconnect))
        cutoff = at
    result = executor.main()
    assert result == (0 if failure in ("operator", "eof") else 2)
    assert any(
        any(v) for t, topic, v in runtime.publications if topic == executor.ARM_TOPIC
    )
    assert all(
        not any(v)
        for t, topic, v in runtime.publications
        if topic == executor.ARM_TOPIC and t >= cutoff
    )
    if result == 2:
        assert any(e["event"] == "fault" for e in runtime.events())


@pytest.mark.parametrize(
    "trigger", ["operator", "deadline", "fault", "malformed", "exit"]
)
def test_rollout_rejects_chunks_after_asynchronous_stop(tmp_path, monkeypatch, trigger):
    """A real pipe child substitutes for ROS while the inference caller is idle."""
    import os
    import sys

    from toolkits.realworld_check import fr3_pi05_supervised_rollout as rollout

    child = tmp_path / "executor.py"
    child.write_text("""import json, sys
print(json.dumps({"event": "ready"}), flush=True)
for line in sys.stdin:
    op = json.loads(line)["op"]
    if op == "chunk":
        print(json.dumps({"event": "accepted"}), flush=True)
    elif op == "end_episode":
        print(json.dumps({"event": "episode_ended"}), flush=True)
    elif op == "quit":
        break
    elif op == "test_fault":
        print(json.dumps({"event": "fault", "reason": "simulated ROS failure"}), flush=True)
        break
    elif op == "test_malformed":
        print("invalid json", flush=True)
    elif op == "test_exit":
        break
""")
    monkeypatch.setattr(rollout, "EXECUTOR", child)
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd) as operator_input:
        executor = rollout.Executor(ros_python=sys.executable)
        try:
            executor.wait("ready")
            executor.send("chunk", values=[])
            executor.wait("accepted")
            if trigger in ("operator", "deadline"):
                executor.monitor_stop(
                    time.monotonic() + (0.05 if trigger == "deadline" else 10),
                    operator_input=operator_input,
                )
                if trigger == "operator":
                    os.write(write_fd, b"\n")
            else:
                executor.send("test_" + trigger)
            # No inference-thread polling or chunk_done wait is needed to stop.
            assert executor.stopped.wait(timeout=2)
            with pytest.raises(rollout.RolloutStopped):
                executor.send("chunk", values=[])
            if trigger == "malformed":
                with pytest.raises(RuntimeError, match="Malformed"):
                    executor.wait("episode_ended")
            if trigger in ("operator", "deadline", "malformed"):
                executor.wait("episode_ended")
            assert len([e for e in executor.events if e["event"] == "accepted"]) == 1
        finally:
            executor.close()
            os.close(write_fd)


def test_real_loop_restarts_only_with_a_fresh_chunk_after_neutral_gap(ros_runtime):
    runtime, executor = ros_runtime
    runtime.pending = [(t, cb) for t, cb in runtime.pending if t < 4_000_000_000]
    runtime.pending.extend(
        [
            (3_516_000_000, runtime.chunk),
            (4_700_000_000, lambda: runtime.send("quit")),
        ]
    )
    assert executor.main() == 0
    actions = [e for e in runtime.events() if e["event"] == "action"]
    assert [e["command_index"] for e in actions] == list(range(10)) * 2
    arm = [
        (t, v) for t, topic, v in runtime.publications if topic == executor.ARM_TOPIC
    ]
    assert all(not any(v) for t, v in arm if 3_032_000_000 <= t < 3_516_000_000)
    assert any(any(v) for t, v in arm if t == 3_516_000_000)


@pytest.mark.parametrize(
    "age_field,age_ns", [("observation_ns", 1_000_000_001), ("issued_ns", 300_000_001)]
)
def test_real_loop_stale_chunk_faults_before_motion(ros_runtime, age_field, age_ns):
    runtime, executor = ros_runtime
    runtime.pending = [(t, cb) for t, cb in runtime.pending if t != 2_032_000_000]

    def stale_chunk():
        stamps = {"observation_ns": runtime.ns, "issued_ns": runtime.ns}
        stamps[age_field] -= age_ns
        if age_field == "issued_ns":
            stamps["observation_ns"] = stamps["issued_ns"]
        runtime.send("chunk", values=[[0.4] * 7] * 10, **stamps)

    runtime.pending.append((2_032_000_000, stale_chunk))
    assert executor.main() == 2
    assert all(
        not any(v)
        for t, topic, v in runtime.publications
        if topic == executor.ARM_TOPIC
    )
    assert any(
        e["event"] == "fault" and "Stale" in e["reason"] for e in runtime.events()
    )


@pytest.mark.parametrize("oldest_age_ns", [900_000_000, 1_100_000_000])
def test_oldest_sensor_receipt_is_not_refreshed_after_inference(oldest_age_ns):
    from toolkits.realworld_check.fr3_pi05_supervised_rollout import (
        oldest_observation_ns,
    )

    now = time.monotonic_ns()
    multimodal = {
        "metadata": {
            "source": {
                name: {"received_monotonic_ns": now}
                for name in (
                    "joint_state",
                    "tcp_pose",
                    "external_wrench",
                    "gripper_state",
                )
            },
            "cameras": {
                role: {"received_monotonic_ns": now} for role in ("base", "wrist")
            },
        }
    }
    multimodal["metadata"]["source"]["gripper_state"]["received_monotonic_ns"] -= (
        oldest_age_ns
    )
    stamp = oldest_observation_ns(multimodal)
    assert stamp == now - oldest_age_ns
    if oldest_age_ns > 1_000_000_000:
        with pytest.raises(ValueError, match="Stale"):
            validate_chunk(
                [[0.0] * 7] * 10, observation_ns=stamp, issued_ns=now, now_ns=now
            )
    else:
        assert validate_chunk(
            [[0.0] * 7] * 10, observation_ns=stamp, issued_ns=now, now_ns=now
        )


def test_measured_travel_limit_still_faults_before_further_motion(ros_runtime):
    runtime, executor = ros_runtime
    runtime.pending.append((2_400_000_000, lambda: setattr(runtime, "pose_x", 0.081)))
    assert executor.main() == 2
    assert any("8 cm" in event.get("reason", "") for event in runtime.events())
    assert all(
        not any(v)
        for t, topic, v in runtime.publications
        if topic == executor.ARM_TOPIC and t > 2_400_000_000
    )


def test_episode_watchdog_still_ends_neutral_wait(ros_runtime):
    runtime, executor = ros_runtime
    runtime.pending = [(t, cb) for t, cb in runtime.pending if t < 4_000_000_000]
    runtime.pending.append((24_000_000_000, lambda: runtime.send("quit")))
    assert executor.main() == 2
    assert any("time limit" in event.get("reason", "") for event in runtime.events())
    assert runtime.ns < 23_000_000_000
