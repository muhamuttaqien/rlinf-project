"""Offline-only request transform tests; no ROS or robot."""
import numpy as np
import pytest

from openpi.models.model import ModelType
from rlinf.models.embodiment.openpi.policies.fr3_peg_request_policy import (
    Fr3PegRequestInputs, Fr3PegRequestOutputs,
)


def _sample():
    state = np.zeros(19, np.float32)
    state[6] = 1
    return {
        "observation/image": np.zeros((64, 64, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((3, 32, 32), dtype=np.uint8),
        "observation/state": state,
        "actions": np.column_stack([
            np.zeros((10, 6), dtype=np.float32),
            np.ones(10, dtype=np.float32),
        ]),
        "prompt": "Insert peg",
    }


def test_request_transform_preserves_action_semantics():
    result = Fr3PegRequestInputs(32, ModelType.PI05)(_sample())
    assert result["state"].shape == (32,)
    assert result["actions"].shape == (10, 32)
    assert np.all(result["actions"][:, 6] == 1)
    assert np.all(result["actions"][:, 7:] == 0)
    assert result["image"]["base_0_rgb"].shape == (64, 64, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (32, 32, 3)
    assert bool(result["image_mask"]["right_wrist_0_rgb"]) is False
    sliced = Fr3PegRequestOutputs()({"actions": result["actions"]})
    assert sliced["actions"].shape == (10, 7)


def test_rejects_metric_action_and_nonbinary_gripper():
    bad = _sample()
    bad["actions"][0, 0] = 1.5
    with pytest.raises(ValueError, match="normalized"):
        Fr3PegRequestInputs()(bad)
    bad = _sample()
    bad["actions"][0, 6] = 0.5
    with pytest.raises(ValueError, match="normalized"):
        Fr3PegRequestInputs()(bad)
