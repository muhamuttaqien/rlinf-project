"""Offline-only tests for two-view FR3 Pi0.5 proxy tensor mapping."""

import numpy as np
import pytest

from openpi.models import model as _model
from rlinf.models.embodiment.openpi.policies.fr3_peg_proxy_policy import (
    Fr3PegProxyInputs, Fr3PegProxyOutputs,
)


def _data():
    return {
        "observation/image": np.zeros((3, 480, 640), dtype=np.uint8),
        "observation/wrist_image": np.ones((3, 480, 848), dtype=np.uint8),
        "observation/state": np.arange(19, dtype=np.float32),
        "actions": np.array([[0.01, 0, 0, 0, 0, 0, .2]], dtype=np.float32),
        "prompt": "Insert the peg.",
    }


def test_both_camera_views_and_actions_are_preserved():
    out = Fr3PegProxyInputs(action_dim=32)(_data())
    assert out["image"]["base_0_rgb"].shape == (480, 640, 3)
    assert out["image"]["left_wrist_0_rgb"].shape == (480, 848, 3)
    assert out["image"]["right_wrist_0_rgb"].shape == (480, 640, 3)
    assert list(out["image_mask"].values()) == [True, True, False]
    np.testing.assert_array_equal(out["state"][:19], np.arange(19))
    assert out["state"].shape == (32,)
    assert out["actions"].shape == (1, 32)
    np.testing.assert_allclose(out["actions"][0, :7], _data()["actions"][0])


def test_extra_action_dimensions_are_only_padding():
    out = Fr3PegProxyInputs(action_dim=32)(_data())
    np.testing.assert_array_equal(out["actions"][0, 7:], np.zeros(25))
    restored = Fr3PegProxyOutputs()({"actions": out["actions"]})
    assert restored["actions"].shape == (1, 7)


def test_missing_wrist_image_fails_not_silent_mask():
    data = _data()
    del data["observation/wrist_image"]
    with pytest.raises(KeyError):
        Fr3PegProxyInputs()(data)


def test_wrong_action_dimension_fails():
    data = _data()
    data["actions"] = np.zeros((1, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="7D"):
        Fr3PegProxyInputs()(data)


def test_wrong_model_type_is_refused():
    with pytest.raises(ValueError, match="PI0.5"):
        Fr3PegProxyInputs(model_type=_model.ModelType.PI0)(_data())
