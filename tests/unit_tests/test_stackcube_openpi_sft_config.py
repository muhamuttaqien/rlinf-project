# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contract tests for StackCube Panda joint-space OpenPI SFT config."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = (
    REPO_ROOT
    / "examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05.yaml"
)
EXPECTED_TASK = "stack the red cube on top of the green cube and release it"


def test_stackcube_sft_yaml_matches_collector():
    cfg = OmegaConf.load(RECIPE)
    assert cfg.runner.task_type == "sft"
    assert cfg.actor.model.action_dim == 8
    assert cfg.actor.model.num_action_chunks == 10
    assert cfg.actor.model.openpi.config_name == "pi05_stackcube_panda_joint"
    assert cfg.actor.model.openpi.action_horizon == 10
    assert cfg.actor.model.openpi.num_images_in_input == 1
    assert cfg.actor.model.openpi.discrete_state_input is False
    assert cfg.actor.model.openpi.use_rlt is False
    assert cfg.actor.fsdp_config.sharding_strategy == "no_shard"


def test_stackcube_openpi_transform_ignores_placeholder_wrist():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.policies.maniskill_policy import (
        ManiSkillInputs,
        ManiSkillOutputs,
    )

    cfg = get_openpi_config("pi05_stackcube_panda_joint")
    assert cfg.model.action_horizon == 10
    assert cfg.model.discrete_state_input is False
    assert cfg.data.output_action_dim == 8
    assert cfg.data.use_wrist_image is False
    assert cfg.data.default_prompt == EXPECTED_TASK

    convert = ManiSkillInputs(
        model_type=cfg.model.model_type,
        use_wrist_image=cfg.data.use_wrist_image,
        default_prompt=cfg.data.default_prompt,
    )
    base = np.full((224, 224, 3), 127, dtype=np.uint8)
    fake_wrist = np.full((224, 224, 3), 255, dtype=np.uint8)
    example = convert(
        {
            "observation/image": base,
            "observation/wrist_image": fake_wrist,
            "observation/state": np.arange(9, dtype=np.float32),
            "actions": np.zeros((10, 8), dtype=np.float32),
        }
    )
    assert example["state"].shape == (9,)
    assert example["prompt"] == EXPECTED_TASK
    assert example["image_mask"]["base_0_rgb"]
    assert not example["image_mask"]["left_wrist_0_rgb"]
    assert example["actions"].shape == (10, 8)
    assert np.all(example["image"]["left_wrist_0_rgb"] == 0)

    outputs = ManiSkillOutputs(output_action_dim=8)
    predicted = outputs({"actions": np.zeros((10, 32), dtype=np.float32)})
    assert predicted["actions"].shape == (10, 8)


def test_v3_norm_stats_streams_parquet_shards_without_v2_metadata(tmp_path):
    """V3 normalization must not use the old JSONL-only LeRobot loader."""
    import importlib.util
    import json

    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("openpi")

    script = REPO_ROOT / "toolkits/lerobot/calculate_norm_stats.py"
    spec = importlib.util.spec_from_file_location(
        "stackcube_calculate_norm_stats", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    root = tmp_path / "dataset"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_frames": 4,
                "features": {
                    "state": {"shape": [9]},
                    "actions": {"shape": [8]},
                },
            }
        ),
        encoding="utf-8",
    )
    for index, offset in enumerate((0, 2)):
        directory = root / "data" / "chunk-000"
        directory.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "state": [float(v)] * 9,
                "actions": [float(v * 2)] * 8,
            }
            for v in range(offset, offset + 2)
        ]
        pq.write_table(
            pyarrow.Table.from_pylist(rows),
            directory / f"file-{index:03d}.parquet",
        )

    stats = module.compute_v3_lerobot_norm_stats(root)
    assert set(stats) == {"state", "actions"}
    np.testing.assert_allclose(stats["state"].mean, np.full(9, 1.5))
    np.testing.assert_allclose(stats["actions"].mean, np.full(8, 3.0))
    assert stats["state"].std.shape == (9,)
    assert stats["actions"].q99.shape == (8,)

    (meta / "info.json").write_text(
        json.dumps(
            {
                "total_frames": 5,
                "features": {
                    "state": {"shape": [9]},
                    "actions": {"shape": [8]},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frame count mismatch"):
        module.compute_v3_lerobot_norm_stats(root)
