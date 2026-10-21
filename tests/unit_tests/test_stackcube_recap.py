# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""StackCube replay conversion and OpenPI data-interface contracts."""

import json

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.check_stackcube_recap import validate_dataset


@pytest.fixture
def converted_replay(tmp_path):
    """Create real source and converted files with a known correspondence."""
    dataset = tmp_path / "dataset"
    (dataset / "data").mkdir(parents=True)
    (dataset / "meta").mkdir()
    states = np.arange(54, dtype=np.float32).reshape(3, 18) / 10
    actions = np.arange(21, dtype=np.float32).reshape(3, 7) / 20
    pq.write_table(
        pa.table(
            {
                "observation.state": states.tolist(),
                "action": actions.tolist(),
                "episode_index": [0, 0, 0],
                "index": [0, 1, 2],
                "is_success": [[False], [False], [True]],
            }
        ),
        dataset / "data/episode.parquet",
    )
    (dataset / "meta/info.json").write_text(
        json.dumps({"total_episodes": 1, "total_frames": 3, "fps": 20})
    )
    replay = tmp_path / "replay.h5"
    with h5py.File(replay, "w") as handle:
        group = handle.create_group("traj_0")
        group["actions"] = actions
        group["obs/agent/qpos"] = np.vstack([states[:, :9], states[-1, :9]])
        group["obs/agent/qvel"] = np.vstack([states[:, 9:], states[-1, 9:]])
        group["success"] = [False, False, True]
    return replay, dataset, states, actions


def test_stackcube_conversion_and_raw_normalization(converted_replay):
    replay, dataset, states, actions = converted_replay
    report, stats = validate_dataset(replay, dataset)
    assert report["frames"] == 3
    assert report["episodes"][0]["first_success_step"] == 3
    for key, array in (("state", states), ("actions", actions)):
        np.testing.assert_allclose(
            stats["norm_stats"][key]["mean"], array.mean(0), atol=1e-6
        )
        np.testing.assert_allclose(
            stats["norm_stats"][key]["q01"], np.quantile(array, 0.01, axis=0)
        )


def test_stackcube_rejects_misaligned_actions(converted_replay):
    replay, dataset, _, _ = converted_replay
    with h5py.File(replay, "r+") as handle:
        handle["traj_0/actions"][1, 0] = 100
    with pytest.raises(AssertionError):
        validate_dataset(replay, dataset)


def test_stackcube_rejects_wrong_control_frequency(converted_replay):
    replay, dataset, _, _ = converted_replay
    path = dataset / "meta/info.json"
    info = json.loads(path.read_text())
    info["fps"] = 30
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="20 Hz"):
        validate_dataset(replay, dataset)


def test_stackcube_openpi_mapping_and_legacy_defaults(tmp_path):
    pytest.importorskip("openpi")
    import openpi.transforms as transforms
    from openpi.models.pi0_config import Pi0Config

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.dataconfig.maniskill_dataconfig import (
        LeRobotManiSkillDataConfig,
    )

    config = get_openpi_config("pi05_stackcube")
    data = config.data.create(tmp_path, config.model)
    assert data.action_sequence_keys == ("action",)
    assert not data.prompt_from_task
    sample = {
        "observation.images.base_camera": np.full((128, 128, 3), 17, dtype=np.uint8),
        "observation.images.hand_camera": np.full((128, 128, 3), 233, dtype=np.uint8),
        "observation.state": np.arange(18, dtype=np.float32),
        "action": np.zeros((10, 7), dtype=np.float32),
        "task": "Stack the red cube on the green cube",
    }
    result = transforms.compose(
        [*data.repack_transforms.inputs, *data.data_transforms.inputs]
    )(sample)
    assert result["prompt"] == sample["task"]
    np.testing.assert_array_equal(
        result["image"]["base_0_rgb"], sample["observation.images.base_camera"]
    )
    np.testing.assert_array_equal(
        result["image"]["left_wrist_0_rgb"], sample["observation.images.hand_camera"]
    )
    assert result["image_mask"]["left_wrist_0_rgb"]
    assert not result["image_mask"]["right_wrist_0_rgb"]
    legacy = LeRobotManiSkillDataConfig().create(tmp_path, Pi0Config())
    old_sample = {
        "observation.images.top": sample["observation.images.base_camera"],
        "observation.state": sample["observation.state"],
        "actions": sample["action"],
        "prompt": sample["task"],
    }
    result = transforms.compose(
        [*legacy.repack_transforms.inputs, *legacy.data_transforms.inputs]
    )(old_sample)
    assert not result["image_mask"]["left_wrist_0_rgb"]
    np.testing.assert_array_equal(result["actions"], sample["action"])
