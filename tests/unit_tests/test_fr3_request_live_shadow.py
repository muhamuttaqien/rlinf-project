"""Read-only live shadow contracts: no ROS publisher or robot action client."""
import ast
from pathlib import Path

import numpy as np
import pytest

from toolkits.realworld_check.fr3_request_live_shadow import (
    check_decoded_requests, make_live_policy_sample,
)


def sample():
    pose = np.array([0.1, 0.2, 0.3, 0, 0, 0, 1], np.float32)
    return {
        "metadata": {"schema": "fr3_robotiq_multimodal_raw_v1"},
        "state": {
            "tcp_pose": pose,
            "arm_joint_position": np.arange(7, dtype=np.float32),
            "tcp_force": np.array([1, 2, 3], np.float32),
            "robotiq_knuckle_position": np.array([0.25], np.float32),
            "tcp_torque": np.array([3, 4, 0], np.float32),
        },
        "policy_images_rgb": {
            "base": np.array([[[10, 20, 30]]], dtype=np.uint8),
            "wrist": np.array([[[40, 50, 60]]], dtype=np.uint8),
        },
    }


def test_live_request_sample_matches_sft_repacks():
    obs = make_live_policy_sample(sample(), "Insert the peg into the hole.")
    assert set(obs) == {
        "observation/image", "observation/wrist_image",
        "observation/state", "prompt",
    }
    assert obs["observation/state"].shape == (19,)
    np.testing.assert_array_equal(obs["observation/state"][:7], sample()["state"]["tcp_pose"])
    np.testing.assert_array_equal(obs["observation/state"][7:14], np.arange(7))
    np.testing.assert_array_equal(obs["observation/state"][-2:], [0.25, 5])
    np.testing.assert_array_equal(obs["observation/image"][0, 0], [10, 20, 30])


def test_bad_live_sample_is_rejected():
    with pytest.raises(ValueError):
        make_live_policy_sample(sample(), "")
    damaged = sample()
    damaged["state"]["tcp_pose"][3:] = 0
    with pytest.raises(ValueError, match="quaternion"):
        make_live_policy_sample(damaged, "peg")
    damaged = sample()
    damaged["policy_images_rgb"].pop("wrist")
    with pytest.raises(ValueError, match="Two live RGB"):
        make_live_policy_sample(damaged, "peg")


def test_shadow_prediction_is_never_an_executable_action():
    raw = np.zeros((10, 7), dtype=np.float32)
    np.testing.assert_array_equal(check_decoded_requests(raw), raw)
    with pytest.raises(ValueError):
        check_decoded_requests(np.zeros((9, 7)))
    raw[0, 0] = np.nan
    with pytest.raises(ValueError):
        check_decoded_requests(raw)


def test_shadow_runner_has_no_command_capability():
    import toolkits.realworld_check.fr3_request_live_shadow as runner
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    calls = [
        n.func.attr for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    assert not set(calls).intersection({
        "create_publisher", "create_client", "create_timer",
        "publish", "send_goal_async", "send_goal", "move",
        "move_to", "execute", "command", "set_target",
    })
