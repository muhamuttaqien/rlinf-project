"""Read-only checks of measured TCP path and action logging."""
import json

from toolkits.realworld_check.fr3_rollout_motion_diagnostic import diagnose


def test_motion_diagnostic(tmp_path):
    path = tmp_path / "policy_command_log.jsonl"
    rows = [
        {"tcp_xyz": [0.1, 0.2, 0.3], "sent_arm_6d": [0.01, 0, 0, 0, 0, 0],
         "requested_7d": [0.1, 0, 0, 0, 0, 0, 1.0], "gripper_event": "open"},
        {"tcp_xyz": [0.101, 0.2, 0.3], "sent_arm_6d": [0.01, 0, 0, 0, 0, 0],
         "requested_7d": [0.1, 0, 0, 0, 0, 0, 1.0], "gripper_event": "hold"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = diagnose(path)
    assert result["published_arm_requests"] == 2
    assert result["max_tcp_displacement_mm"] == 1.0
    assert result["logged_gripper_events"]["open"] == 1
