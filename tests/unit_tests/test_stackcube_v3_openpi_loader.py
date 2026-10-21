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

"""Regression tests for StackCube v3 embedded-Parquet/OpenPI compatibility."""

import io
import json
import pickle
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
    from PIL import Image

    root = tmp_path / "local" / "stackcube"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

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

    rows = []
    for idx in range(6):
        png_data = io.BytesIO()
        Image.fromarray(
            np.full((224, 224, 3), idx * 30, dtype=np.uint8)
        ).save(png_data, format="PNG")
        rows.append(
            {
                "index": idx,
                "episode_index": idx // 3,
                "frame_index": idx % 3,
                "state": [float(idx)] * 9,
                "actions": [float(idx)] * 8,
                "image": {"bytes": png_data.getvalue(), "path": None},
            }
        )

    # An episode spans two Parquet shards and the first file has 2 row groups.
    pq.write_table(
        pa.Table.from_pylist(rows[:4]),
        root / "data" / "chunk-000" / "file-000.parquet",
        row_group_size=2,
    )
    pq.write_table(
        pa.Table.from_pylist(rows[4:]),
        root / "data" / "chunk-000" / "file-001.parquet",
        row_group_size=1,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "episode_index": ep,
                    "length": 3,
                    "dataset_from_index": ep * 3,
                    "dataset_to_index": (ep + 1) * 3,
                }
                for ep in range(2)
            ]
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    return root


def test_v3_reader_loads_images_from_parquet_without_lerobot(v3_stackcube):
    dataset = V3StackCubeDataset(v3_stackcube, action_horizon=4)
    assert is_lerobot_v3_dataset(v3_stackcube)
    assert len(dataset) == 6
    assert len(dataset.episodes) == 2

    last_first = dataset[2]
    np.testing.assert_allclose(last_first["actions"][:, 0], [2, 2, 2, 2])
    assert last_first["state"].shape == (9,)
    assert last_first["image"].shape == (224, 224, 3)
    assert last_first["image"].dtype == np.uint8
    assert not last_first["wrist_image"].any()
    np.testing.assert_array_equal(last_first["frame_index"], 2)
    assert np.all(last_first["image"] == 60)

    first_second = dataset[3]
    np.testing.assert_allclose(first_second["actions"][:, 0], [3, 4, 5, 5])
    assert first_second["frame_index"] == 0
    assert np.all(first_second["image"] == 90)

    across_shards = dataset[4]
    assert np.all(across_shards["image"] == 120)

    restored = pickle.loads(pickle.dumps(dataset))
    assert np.all(restored[5]["image"] == 150)


def test_v3_reader_rejects_wrong_metadata(v3_stackcube):
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
