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

"""Unit tests: W&B media uses actual LeRobot data and bounded frame sampling."""

import sys
import types

import numpy as np

from rlinf.utils import stackcube_wandb_media as media


def test_chw_float_image_to_rgb_uint8():
    image = np.zeros((3, 2, 4), dtype=np.float32)
    image[0, :, :] = 1.0
    result = media.to_rgb_uint8(image)
    assert result.shape == (2, 4, 3)
    assert result.dtype == np.uint8
    assert (result[..., 0] == 255).all()
    assert (result[..., 1:] == 0).all()


def test_hwc_uint8_preserved():
    frame = np.full((8, 9, 3), 123, dtype=np.uint8)
    np.testing.assert_array_equal(media.to_rgb_uint8(frame), frame)


def test_bounded_episode_sampling_includes_endpoints():
    assert media.evenly_spaced_indices(np.arange(10, 15), 3) == [10, 12, 14]


def test_stackcube_media_contains_images_and_demonstration_videos(
    monkeypatch, tmp_path
):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text("{}", encoding="utf-8")

    class FakeFrames:
        column_names = ["episode_index"]

        def __getitem__(self, column):
            assert column == "episode_index"
            return [0, 0, 0, 0, 1, 1, 1]

    class FakeDataset:
        hf_dataset = FakeFrames()
        fps = 20

        def __init__(self, repo_id, root):
            assert repo_id == "local/stackcube_demo"
            assert root == tmp_path

        def __getitem__(self, index):
            frame = np.full((3, 8, 8), index / 10, dtype=np.float32)
            return {"image": frame}

    class FakeWandb:
        def Image(self, pixels, caption):
            assert pixels.shape == (8, 8, 3)
            return ("image", pixels.copy(), caption)

        def Video(self, frames, fps, format, caption):
            assert frames.ndim == 4 and frames.shape[1] == 3
            assert frames.shape[0] <= 3
            assert frames.dtype == np.uint8
            assert format == "mp4"
            return ("video", frames.copy(), fps, caption)

    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = FakeDataset
    monkeypatch.setitem(sys.modules, "lerobot", types.ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", types.ModuleType("lerobot.datasets"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)
    monkeypatch.setattr(media, "resolve_lerobot_dataset_root", lambda _: tmp_path)

    output = media.build_stackcube_wandb_media(
        "local/stackcube_demo", FakeWandb(), max_episodes=2, frames_per_video=3
    )

    assert len(output) == 4
    for ep in (0, 1):
        key = f"dataset/expert_episode_{ep:03d}"
        assert len(output[f"{key}/snapshots"]) == 3
        assert output[f"{key}/demonstration_video"][0] == "video"
