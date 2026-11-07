# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""SIM-6: keep expert positives and preserve labeled rollout negatives."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from rlinf.data.datasets.recap.cfg_model import (
    AdvantagePreservingDataset,
    CfgMixtureDataset,
)


class TinyDataset:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return dict(self.values[idx])


class TinyHF:
    def __init__(self):
        self.column_names = ("episode_index", "frame_index")
        self.columns = {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
        }

    def __getitem__(self, key):
        return self.columns[key]

    def __len__(self):
        return 2


def test_sft_requires_no_sidecar_or_video_metadata():
    data = AdvantagePreservingDataset(
        base_dataset=None,
        transformed_dataset=TinyDataset([{"actions": 1}, {"actions": 2}]),
        force_positive=True,
    )
    assert [data[i]["advantage"] for i in range(len(data))] == [True, True]


def test_rollout_retains_positive_and_negative_labels():
    base = SimpleNamespace(hf_dataset=TinyHF())
    data = AdvantagePreservingDataset(
        base_dataset=base,
        transformed_dataset=TinyDataset([{"actions": 1}, {"actions": 2}]),
        advantages_lookup={(0, 0): True, (0, 1): False},
    )
    assert [data[i]["advantage"] for i in range(len(data))] == [True, False]


def test_mixture_balances_small_sft_and_large_rollout():
    mix = CfgMixtureDataset(
        [
            (TinyDataset([{"advantage": True}]), 1.0),
            (TinyDataset([{"advantage": False}] * 100), 1.0),
        ],
        balance_dataset_weights=False,
    )
    np.testing.assert_allclose(mix.dataset_sampling_weights, [0.5, 0.5])


def test_sim6_config_points_to_both_sources():
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(
        root
        / "examples/offline_rl/config/experiments/libero10_task0/cfg_rl_openpi.yaml"
    )
    sources = cfg.data.train_data_paths
    assert [source.type for source in sources] == ["sft", "rollout"]
    assert [source.weight for source in sources] == [1.0, 1.0]
    assert cfg.data.balance_dataset_weights is False
    assert cfg.data.advantage_tag == "libero10_task0_recap"
    assert cfg.actor.model.openpi.config_name == "pi05_libero"
    assert "pi05_base_rlinf" in cfg.actor.model.model_path
