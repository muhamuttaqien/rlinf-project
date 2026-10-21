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

"""Contract tests for read-only embedded-image LeRobot v3 OpenPI integration."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.data.datasets.openpi.v3_stackcube_dataset import (
    V3StackCubeDataset,
    is_lerobot_v3_dataset,
)


@pytest.fixture
def v3_stackcube(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    root = tmp_path / "local" / "stackcube"
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 20,
        "total_frames": 6,
        "total_episodes": 2,
        "features": {
            "state": {"shape": [9], "dtype": "float32"},
            "actions": {"shape": [8], "dtype": "float32"},
            "image": {"shape": [224, 224, 3], "dtype": "image"},
            "wrist_image": {"shape": [224, 224, 3], "dtype": "image"},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    pq.write_table(
        pa.Table.from_pylist(
            [{"task_index": 0, "__index_level_0__": "stack the red cube"}]
        ),
        root / "meta" / "tasks.parquet",
    )
    return root


def test_v3_reader_uses_native_lerobot_for_embedded_images_and_action_windows(
    v3_stackcube, monkeypatch
):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    called = {}
    class FakeNativeDataset:
        def __init__(self, repo_id, root, delta_timestamps):
            called.update(
                repo_id=repo_id,
                root=root,
                delta_timestamps=delta_timestamps,
            )
            self.meta = SimpleNamespace(episodes=[{}, {}])

        def __len__(self):
            return 6

        def __getitem__(self, idx):
            episode_end = ((idx // 3) + 1) * 3 - 1
            actions = np.asarray(
                [[min(idx + j, episode_end)] * 8 for j in range(4)],
                dtype=np.float32,
            )
            return {
                "state": np.full(9, idx, dtype=np.float32),
                "actions": actions,
                "image": np.ones((3, 224, 224), dtype=np.float32) / 2,
                "episode_index": idx // 3,
                "frame_index": idx % 3,
            }

    monkeypatch.setattr(lerobot_dataset, "LeRobotDataset", FakeNativeDataset)
    dataset = V3StackCubeDataset(v3_stackcube, action_horizon=4)
    assert is_lerobot_v3_dataset(v3_stackcube)
    assert len(dataset) == 6
    assert called["root"] == v3_stackcube.resolve()
    np.testing.assert_allclose(
        called["delta_timestamps"]["actions"], np.arange(4) / 20
    )

    last_first = dataset[2]
    np.testing.assert_allclose(last_first["actions"][:, 0], [2, 2, 2, 2])
    assert last_first["image"].shape == (3, 224, 224)
    assert last_first["state"].shape == (9,)
    assert not last_first["wrist_image"].any()

    first_second = dataset[3]
    np.testing.assert_allclose(first_second["actions"][:, 0], [3, 4, 5, 5])
    assert first_second["frame_index"] == 0


def test_v3_reader_rejects_wrong_metadata(v3_stackcube, monkeypatch):
    info_path = v3_stackcube / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["actions"]["shape"] = [7]
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected StackCube actions"):
        V3StackCubeDataset(v3_stackcube, action_horizon=10)


def test_v3_factory_override_is_restored_after_loader_creation(tmp_path):
    pytest.importorskip("openpi")
    from rlinf.data.datasets.openpi.official_sft_data_loader import (
        _create_openpi_v3_data_loader,
    )

    def original_factory(*args):
        raise AssertionError("v2 factory should not be called")

    def fake_loader(*args, **kwargs):
        del args, kwargs
        assert fake_module.create_torch_dataset is not original_factory
        return "data loader"

    fake_module = SimpleNamespace(
        create_torch_dataset=original_factory,
        create_data_loader=fake_loader,
    )
    assert (
        _create_openpi_v3_data_loader(
            fake_module, SimpleNamespace(), tmp_path, shuffle=False
        )
        == "data loader"
    )
    assert fake_module.create_torch_dataset is original_factory
