"""Parser-only FR3 command interface audit tests; never invoke ROS."""

from toolkits.realworld_check.fr3_command_interface_audit import (
    discovery_report, parse_actions, parse_controller_timeout, parse_topic_info,
)

TOPIC = """Type: geometry_msgs/msg/Twist
Publisher count: 1
Node name: spacemouse_publisher
Node namespace: /
Endpoint type: PUBLISHER
Subscription count: 1
Node name: joint_impedance_ik_controller
Node namespace: /
Endpoint type: SUBSCRIPTION
"""


def test_parses_ros2_topic():
    t = parse_topic_info(TOPIC)
    assert t["message_type"] == "geometry_msgs/msg/Twist"
    assert t["publishers"] == 1
    assert t["subscriptions"] == 1
    assert "joint_impedance_ik_controller" in t["reported_nodes"]


def test_parses_param_and_action():
    assert parse_controller_timeout("Double value is: 0.25") == 0.25
    assert parse_actions(
        "/robotiq_gripper_controller/gripper_cmd [control_msgs/action/GripperCommand]"
    ) == {"/robotiq_gripper_controller/gripper_cmd":
          "control_msgs/action/GripperCommand"}


def test_report_still_blocks_actuation_after_successful_discovery():
    report = discovery_report(
        TOPIC, "Double value is: 0.25",
        "/robotiq_gripper_controller/gripper_cmd [control_msgs/action/GripperCommand]"
    )
    assert report["graph_discovery_complete"] is True
    assert report["arm_topic_type_matches"] is True
    assert report["gripper_action_type_matches"] is True
    assert report["physical_execution_approved"] is False
    assert report["policy_action_mapping_verified"] is False
    assert report["policy_command_bridge_present"] is False
    assert report["robot_commands_sent"] is False


def test_missing_timeout_cannot_pass():
    report = discovery_report(TOPIC, "", "")
    assert report["graph_discovery_complete"] is False
    assert report["gripper_action_type_matches"] is False


def test_query_failure_is_not_readiness():
    report = discovery_report(
        TOPIC, "Double value is: 0.25", "",
        command_errors={"gripper_actions": "ROS domain 71 query timed out"},
    )
    assert report["graph_discovery_complete"] is False
    assert report["physical_execution_approved"] is False

def test_ssh_execution_sources_nuc_humble_only(monkeypatch):
    """Remote discovery executes only one ROS CLI read-only query."""
    from toolkits.realworld_check.fr3_command_interface_audit import _read_only_query

    captured = {}

    def stub_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

        class Result:
            returncode = 0
            stdout = "Type: geometry_msgs/msg/Twist\\n"
            stderr = ""
        return Result()

    monkeypatch.setattr(
        "toolkits.realworld_check.fr3_command_interface_audit.subprocess.run",
        stub_run,
    )
    result = _read_only_query(
        ["ros2", "topic", "info", "-v",
         "/franka_controller/target_cartesian_velocity_percent"],
        70, 20, ssh_host="nuc-franka",
    )
    assert result.startswith("Type:")
    assert captured["args"][-3:] == ["nuc-franka", "bash", "-s"]
    sent = captured["kwargs"]["input"]
    assert "source /opt/ros/humble/setup.bash" in sent
    assert "export ROS_DOMAIN_ID=70" in sent
    assert "ros2 topic info -v" in sent
    assert "ros2 topic pub" not in sent
