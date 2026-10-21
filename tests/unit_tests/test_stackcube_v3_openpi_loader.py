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

"""Regression checks for the non-mutating LeRobot v3 StackCube OpenPI adapter."""

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
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "image" / "chunk-000").mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "fps": 20,
        "total_frames": 6,
        "total_episodes": 2,
        "features": {
            "state": {"shape": [9], "dtype": "float32"},
            "actions": {"shape": [8], "dtype": "float32"},
            "image": {"shape": [16, 16, 3], "dtype": "video"},
        },
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    pq.write_table(
        pa.Table.from_pylist(
            [{"task_index": 0, "__index_level_0__": "stack the red cube"}]
        ),
        root / "meta" / "tasks.parquet",
    )

    frames = [
        {
            "index": idx,
            "episode_index": idx // 3,
            "frame_index": idx % 3,
            "timestamp": (idx % 3) / 20,
            "state": [float(idx)] * 9,
            "actions": [float(idx)] * 8,
        }
        for idx in range(6)
    ]
    pq.write_table(
        pa.Table.from_pylist(frames),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    episodes = [
        {
            "episode_index": ep,
            "length": 3,
            "dataset_from_index": ep * 3,
            "dataset_to_index": (ep + 1) * 3,
            "videos/image/chunk_index": 0,
            "videos/image/file_index": 0,
            "videos/image/from_timestamp": ep * 3 / 20,
        }
        for ep in range(2)
    ]
    pq.write_table(
        pa.Table.from_pylist(episodes),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    return root


def test_v3_action_sequences_never_cross_episode_boundaries(v3_stackcube, monkeypatch):
    dataset = V3StackCubeDataset(v3_stackcube, action_horizon=4)
    assert is_lerobot_v3_dataset(v3_stackcube)
    assert len(dataset) == 6
    monkeypatch.setattr(
        V3StackCubeDataset,
        "_read_image",
        lambda self, frame: np.zeros((16, 16, 3), dtype=np.uint8),
    )

    final_first_episode = dataset[2]
    np.testing.assert_allclose(
        final_first_episode["actions"][:, 0], [2, 2, 2, 2]
    )
    assert final_first_episode["state"].shape == (9,)
    assert final_first_episode["wrist_image"].shape == (16, 16, 3)

    first_second_episode = dataset[3]
    np.testing.assert_allclose(
        first_second_episode["actions"][:, 0], [3, 4, 5, 5]
    )
    np.testing.assert_array_equal(first_second_episode["frame_index"], 0)


def test_v3_video_timestamp_uses_episode_offset(v3_stackcube):
    cv2 = pytest.importorskip("cv2")
    path = (
        v3_stackcube
        / "videos"
        / "image"
        / "chunk-000"
        / "file-000.mp4"
    )
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (16, 16)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v encoder is unavailable")
    try:
        for value in (10, 30, 50, 70, 90, 110):
            writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    finally:
        writer.release()

    dataset = V3StackCubeDataset(v3_stackcube, action_horizon=4)
    image = dataset[4]["image"]
    assert image.shape == (16, 16, 3)
    assert np.isfinite(image).all()
    assert abs(float(image.mean()) - 90) < 15


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
    config = SimpleNamespace()
    assert (
        _create_openpi_v3_data_loader(
            fake_module, config, tmp_path, shuffle=False
        )
        == "data loader"
    )
    assert fake_module.create_torch_dataset is original_factory
