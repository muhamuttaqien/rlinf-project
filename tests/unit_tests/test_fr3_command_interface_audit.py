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
