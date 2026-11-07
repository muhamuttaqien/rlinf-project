# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Focused tests for grouped LeRobot video access during RECAP advantages."""

from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Subset

from examples.offline_rl.advantage_labeling.recap.process.compute_advantages import (
    ValueInferenceDataset,
)


class FakeHF:
    def __init__(self):
        self.rows = [
            {
                "episode_index": torch.tensor(i // 2),
                "frame_index": torch.tensor(i % 2),
                "timestamp": torch.tensor((i % 2) * 0.1),
                "task_index": torch.tensor(i // 2),
                "observation.state": torch.tensor([float(i)]),
            }
            for i in range(4)
        ]

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return {k: [self.rows[i][k] for i in idx] for k in self.rows[0]}
        return dict(self.rows[idx])


class FakeLeRobot:
    def __init__(self):
        self.hf_dataset = FakeHF()
        self.meta = SimpleNamespace(
            video_keys=["observation.image", "observation.wrist_image"],
            tasks={0: "first task", 1: "second task"},
        )
        self.delta_indices = None
        self.image_transforms = None
        self.video_calls = []

    def __len__(self):
        return 4

    def _query_videos(self, query_timestamps, ep_idx):
        self.video_calls.append((ep_idx, tuple(query_timestamps["observation.image"])))
        return {
            key: torch.stack([
                torch.full((3, 2, 2), float(ep_idx + ts + cam_idx))
                for ts in timestamps
            ]).squeeze(0)
            for cam_idx, (key, timestamps) in enumerate(query_timestamps.items())
        }

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())
        ts = float(item["timestamp"].item())
        videos = self._query_videos(
            {key: [ts] for key in self.meta.video_keys}, ep_idx
        )
        item = {**videos, **item}
        item["task"] = self.meta.tasks[int(item["task_index"].item())]
        return item


def make_dataset(fake, enabled=True):
    sidecar = {
        ep: {
            "return": np.array([-2.0, -1.0], dtype=np.float32),
            "reward": np.array([0.0, 1.0], dtype=np.float32),
        }
        for ep in (0, 1)
    }
    return ValueInferenceDataset(
        fake, "libero", {}, returns_sidecar=sidecar,
        batch_video_decode=enabled,
    )


def assert_items_equal(expected, actual):
    for left, right in zip(expected, actual):
        assert left.keys() == right.keys()
        for key in left:
            if key == "obs":
                assert left[key].keys() == right[key].keys()
                for obs_key in left[key]:
                    if isinstance(left[key][obs_key], np.ndarray):
                        np.testing.assert_array_equal(
                            left[key][obs_key], right[key][obs_key]
                        )
                    else:
                        assert left[key][obs_key] == right[key][obs_key]
            else:
                assert left[key] == right[key]


def test_batch_decoding_preserves_scalar_dataset_results_and_order():
    fake = FakeLeRobot()
    dataset = make_dataset(fake)
    indices = [2, 0, 3, 1]  # Cross two episode boundaries, out of order.
    reference = [dataset[i] for i in indices]
    fake.video_calls.clear()

    actual = dataset.__getitems__(indices)

    assert_items_equal(reference, actual)
    assert len(fake.video_calls) == 2  # One decode call per episode.
    assert [ep for ep, _ in fake.video_calls] == [1, 0]


def test_single_frame_batch_and_empty_batch():
    fake = FakeLeRobot()
    dataset = make_dataset(fake)
    assert_items_equal([dataset[1]], dataset.__getitems__([1]))
    assert dataset.__getitems__([]) == []


def test_subset_uses_batch_fetch():
    fake = FakeLeRobot()
    dataset = make_dataset(fake)
    subset = Subset(dataset, [2, 0, 3, 1])
    fake.video_calls.clear()

    actual = subset.__getitems__([0, 1, 2, 3])

    assert [item["global_idx"] for item in actual] == [2, 0, 3, 1]
    assert len(fake.video_calls) == 2


def test_batch_mode_can_be_disabled():
    fake = FakeLeRobot()
    dataset = make_dataset(fake, enabled=False)

    actual = dataset.__getitems__([0, 1, 2, 3])

    assert len(actual) == 4
    assert len(fake.video_calls) == 4


def test_temporal_windows_use_original_scalar_path():
    fake = FakeLeRobot()
    fake.delta_indices = {"observation.state": [0, 1]}
    dataset = make_dataset(fake)

    actual = dataset.__getitems__([0, 1])

    assert len(actual) == 2
    assert len(fake.video_calls) == 2
