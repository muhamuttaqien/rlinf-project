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

"""Synthetic FR3 training batches through native dataset and SFT interfaces."""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.data.datasets.openpi.fr3 import FR3OfflineDataset
from rlinf.data.datasets.openpi.fr3_contract import (
    load_fr3_contract,
    normalization_fixture_check,
)
from rlinf.data.datasets.openpi.fr3_training import (
    build_fr3_request_target_proposal,
    load_checkpoint_candidate,
    require_fr3_training_ready,
    tokenize_fr3_fixture,
)
from rlinf.data.storage.fr3_validation import require
from rlinf.models.embodiment.openpi.modules.model import Observation

TRAINING_SCHEMA = "fr3_training_batch_v1"
ACTION_FIELDS = [
    "offset_x",
    "offset_y",
    "offset_z",
    "offset_roll",
    "offset_pitch",
    "offset_yaw",
    "closure",
]


def _identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def synthetic_training_fixture(dataset: FR3OfflineDataset) -> dict:
    """Declare a separate task/split fixture; never change archived annotations."""
    episodes = list(dataset.metadata_document()["episodes"])
    return {
        "schema": "fr3_training_fixture_v1",
        "scope": "synthetic_math_only",
        "source_manifest_sha256": dataset.source_manifest_sha256,
        "task": {
            "task_id": "synthetic:loss_schema",
            "version": "1",
            "instruction": "synthetic probe",
            "origin": "synthetic_fixture",
            "provenance": "literal_offline_schema_test_v1",
        },
        "partitions": {
            r["episode_id"]: "train" if i == 0 else "validation"
            for i, r in enumerate(episodes)
        },
    }


def synthetic_normalization_assets(
    dataset: FR3OfflineDataset, fixture: dict, method: str = "zscore"
) -> dict:
    """Declare constant math-test statistics, with an explicit training-only binding."""
    train = sorted(e for e, split in fixture["partitions"].items() if split == "train")
    return {
        "schema": "fr3_training_normalization_v1",
        "scope": "synthetic_math_only",
        "dataset_id": dataset.manifest["dataset_id"],
        "source_manifest_sha256": dataset.source_manifest_sha256,
        "training_episode_ids": train,
        "training_split_sha256": _identity(train),
        "sample_count": sum(
            e["step_count"] for e in dataset._episodes if e["episode_id"] in train
        ),
        "state_feature_order": dataset.manifest["state_fields"].copy(),
        "action_feature_order": ACTION_FIELDS.copy(),
        "transform_version": "fr3_request_imitation_proposal_v1",
        "checkpoint_candidate_id": load_checkpoint_candidate()["candidate_id"],
        "statistics_provenance": {
            "source_id": "synthetic:constant_statistics",
            "method": "declared_constants_not_estimated",
            "version": "1",
            "timestamp_ns": 1,
            "clock_domain": "synthetic:math_fixture",
        },
        "method": method,
        "statistics": {
            name: {
                "mean": [0.0] * dim,
                "std": [100.0] * dim,
                **(
                    {"q01": [-100.0] * dim, "q99": [100.0] * dim}
                    if method == "quantile"
                    else {}
                ),
            }
            for name, dim in (("state", 42), ("actions", 7))
        },
    }


def validate_training_normalization(
    assets: dict, dataset: FR3OfflineDataset, fixture: dict
) -> dict:
    """Validate statistical identity, order, provenance and train/evaluation isolation."""
    train = sorted(e for e, split in fixture["partitions"].items() if split == "train")
    require(
        assets.get("schema") == "fr3_training_normalization_v1",
        "Unknown FR3 normalization schema",
    )
    require(
        assets.get("dataset_id") == dataset.manifest["dataset_id"]
        and assets.get("source_manifest_sha256") == dataset.source_manifest_sha256,
        "Normalization dataset/manifest identity mismatch",
    )
    require(
        bool(train)
        and assets.get("training_episode_ids") == train
        and assets.get("training_split_sha256") == _identity(train),
        "Normalization training split identity mismatch or evaluation leakage",
    )
    expected_count = sum(
        e["step_count"] for e in dataset._episodes if e["episode_id"] in train
    )
    require(
        type(assets.get("sample_count")) is int
        and assets["sample_count"] == expected_count
        and expected_count > 0,
        "Normalization sample count disagrees with original training frames",
    )
    require(
        assets.get("state_feature_order") == dataset.manifest["state_fields"]
        and assets.get("action_feature_order") == ACTION_FIELDS,
        "Normalization feature order mismatch",
    )
    require(
        assets.get("transform_version") == "fr3_request_imitation_proposal_v1"
        and assets.get("checkpoint_candidate_id")
        == load_checkpoint_candidate()["candidate_id"],
        "Normalization transform/checkpoint association mismatch",
    )
    p = assets.get("statistics_provenance", {})
    require(
        isinstance(p, dict)
        and set(p)
        == {"source_id", "method", "version", "timestamp_ns", "clock_domain"},
        "Missing normalization provenance",
    )
    require(
        all(
            isinstance(p[k], str) and p[k].strip()
            for k in ("source_id", "method", "version", "clock_domain")
        )
        and type(p["timestamp_ns"]) is int
        and p["timestamp_ns"] > 0,
        "Invalid normalization provenance",
    )
    require(
        assets.get("scope") == "synthetic_math_only",
        "UNQUALIFIED: production FR3 statistics require an approved real training dataset/transform",
    )
    require(
        p["method"] == "declared_constants_not_estimated",
        "Synthetic statistics must identify literal math fixtures",
    )
    math_fixture = {
        "schema": "fr3_normalization_fixture_v1",
        "scope": "synthetic_math_only",
        "contract_id": "fr3_openpi_structural_v1",
        "method": assets.get("method"),
        "statistics": assets.get("statistics"),
    }
    normalization_fixture_check(
        np.zeros(42, dtype=np.float32), np.zeros((1, 7), dtype=np.float32), math_fixture
    )
    return math_fixture


@dataclass(frozen=True)
class FR3TrainingConfig:
    """Offline-only schema configuration, distinct from a trainable TrainConfig."""

    assets_root: str | Path
    window_policy: str = "masked_padded_windows"
    partition: str = "train"


class FR3TrainingDataset(Dataset):
    """Prepare fixture-normalized batches while preserving every native sample."""

    def __init__(
        self,
        dataset: FR3OfflineDataset,
        *,
        config: FR3TrainingConfig,
        fixture: dict,
        normalization: dict,
    ) -> None:
        require(
            isinstance(dataset, FR3OfflineDataset),
            "Expected native audited FR3OfflineDataset",
        )
        require(
            config.window_policy in ("masked_padded_windows", "require_full_window"),
            "Unknown FR3 window policy",
        )
        require(
            config.partition in ("train", "validation", "test"),
            "Unknown explicit fixture partition",
        )
        require(
            fixture.get("schema") == "fr3_training_fixture_v1"
            and fixture.get("scope") == "synthetic_math_only"
            and fixture.get("source_manifest_sha256") == dataset.source_manifest_sha256,
            "Unknown or unbound synthetic training fixture",
        )
        task = fixture.get("task", {})
        require(
            set(task) == {"task_id", "version", "instruction", "origin", "provenance"}
            and task["origin"] == "synthetic_fixture"
            and all(isinstance(v, str) and v.strip() for v in task.values()),
            "Missing original instruction; only explicitly attributed synthetic fixture text is allowed",
        )
        ids = {e["episode_id"] for e in dataset._episodes}
        require(
            isinstance(fixture.get("partitions"), dict)
            and set(fixture["partitions"]) == ids
            and all(
                v in ("train", "validation", "test")
                for v in fixture["partitions"].values()
            ),
            "Fixture partitions must assign every original episode exactly once",
        )
        self._math_fixture = copy.deepcopy(
            validate_training_normalization(normalization, dataset, fixture)
        )
        self.native, self.config = dataset, config
        self.fixture, self.normalization = (
            copy.deepcopy(fixture),
            copy.deepcopy(normalization),
        )
        self.indices = []
        self.excluded_full_window_indices = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if (
                fixture["partitions"][sample["provenance"]["episode_id"]]
                != config.partition
            ):
                continue
            if (
                config.window_policy == "require_full_window"
                and sample["actions_is_pad"].any()
            ):
                self.excluded_full_window_indices.append(i)
            else:
                self.indices.append(i)
        require(
            bool(self.indices),
            "No eligible FR3 windows; full-window-only policy cannot pad missing targets",
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        require(
            type(index) is int and 0 <= index < len(self),
            "FR3 training sample index outside dataset",
        )
        source_index = self.indices[index]
        sample = self.native[source_index]
        proposal = build_fr3_request_target_proposal(
            self.native,
            source_index,
            require_full_window=self.config.window_policy == "require_full_window",
        )
        check = normalization_fixture_check(
            sample["native_state"], proposal["recorded_requests"], self._math_fixture
        )
        state, requests = check["normalized"]["state"], check["normalized"]["actions"]
        tokens = tokenize_fr3_fixture(
            self.config.assets_root,
            state,
            self.fixture["task"]["instruction"],
            scope="synthetic_math_only",
        )
        from openpi.transforms import PadStatesAndActions

        padded = PadStatesAndActions(32)(
            {"state": state.copy(), "actions": requests.copy()}
        )
        channels = np.arange(32) < 7
        future = ~sample["actions_is_pad"]
        return {
            "training_schema": TRAINING_SCHEMA,
            "offline_schema_probe": True,
            "action_loss_mask_required": True,
            "physical_execution": False,
            "state": padded["state"],
            "native_state": sample["native_state"],
            "actions": padded["actions"],
            "recorded_requests": proposal["recorded_requests"],
            "valid_action_channels": channels,
            "valid_future_steps": future,
            "action_loss_mask": future[:, None] & channels[None, :],
            "image": sample["image"],
            "image_mask": sample["image_mask"],
            "tokenized_prompt": tokens["tokens"],
            "tokenized_prompt_mask": tokens["mask"],
            "task": copy.deepcopy(self.fixture["task"]),
            "partition": self.config.partition,
            "target_semantics": "synthetic_normalized_request_imitation_proposal",
            "normalization_scope": "synthetic_math_only",
            "token_truncated": False,
            "step_period_ns": self.native.manifest["step_period_ns"],
            "provenance": sample["provenance"],
            "model_training_target": None,
        }

    def require_training_ready(self) -> None:
        require_fr3_training_ready()


def collate_fr3_training_items(items: list[dict]) -> dict:
    """Collate masks/observations/provenance into the native dictionary SFT API."""
    require(bool(items), "Cannot collate an empty FR3 batch")
    image_keys = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    horizon = items[0]["actions"].shape[0]
    for item in items:
        require(
            item.get("task", {}).get("origin") == "synthetic_fixture"
            and bool(item.get("task", {}).get("instruction", "").strip()),
            "Missing explicit synthetic task instruction",
        )
        require(
            {
                "action_loss_mask",
                "valid_action_channels",
                "valid_future_steps",
                "tokenized_prompt",
                "tokenized_prompt_mask",
            }.issubset(item),
            "Missing FR3 masks or token fields",
        )
        require(
            item["provenance"]["state_fields"]
            == [f["name"] for f in load_fr3_contract()["state"]["fields"]],
            "Incorrect FR3 feature ordering",
        )
        indices = np.asarray(item["provenance"]["source_frame_indices"])
        times = np.asarray(item["provenance"]["source_timestamps_ns"])
        require(
            indices.shape == times.shape == (horizon,)
            and indices[0] == item["provenance"]["frame_index"]
            and np.isin(np.diff(indices), [0, 1]).all()
            and (np.diff(times) == np.diff(indices) * item["step_period_ns"]).all(),
            "Incorrect FR3 source frame/timestamp association",
        )
        expected_future = np.r_[True, np.diff(indices) == 1]
        require(
            np.array_equal(item["valid_future_steps"], expected_future)
            and (np.diff(expected_future.astype(int)) <= 0).all(),
            "Future mask disagrees with source frames",
        )
        require(
            item.get("training_schema") == TRAINING_SCHEMA
            and item.get("offline_schema_probe") is True
            and item.get("physical_execution") is False
            and item.get("model_training_target") is None,
            "Unqualified FR3 training schema, checkpoint target or physical claim",
        )
        require(
            item.get("target_semantics")
            == "synthetic_normalized_request_imitation_proposal"
            and item.get("normalization_scope") == "synthetic_math_only",
            "Wrong FR3 action semantics or statistical scope",
        )
        for key, shape in (
            ("state", (42,)),
            ("native_state", (42,)),
            ("actions", (horizon, 32)),
        ):
            a = np.asarray(item[key])
            require(
                a.shape == shape and a.dtype == np.float32 and np.isfinite(a).all(),
                f"Invalid FR3 {key} shape/dtype/values",
            )
        channels, future, mask = (
            np.asarray(item[k])
            for k in ("valid_action_channels", "valid_future_steps", "action_loss_mask")
        )
        require(
            channels.dtype == future.dtype == mask.dtype == np.bool_
            and channels.shape == (32,)
            and future.shape == (horizon,)
            and mask.shape == (horizon, 32)
            and np.array_equal(channels, np.arange(32) < 7)
            and np.array_equal(mask, future[:, None] & channels[None, :])
            and mask.any(),
            "Invalid or missing FR3 action masks",
        )
        require(
            not item["actions"][:, 7:].any(), "Incorrect nonzero 32D action padding"
        )
        require(
            set(item["image"]) == set(item["image_mask"]) == image_keys,
            "Missing FR3 camera slot/mask",
        )
        for k in image_keys:
            a = np.asarray(item["image"][k])
            m = np.asarray(item["image_mask"][k])
            require(
                a.shape == (224, 224, 3)
                and a.dtype == np.float32
                and np.isfinite(a).all()
                and (np.abs(a) <= 1).all()
                and m.shape == ()
                and m.dtype == np.bool_,
                "Invalid prepared FR3 image/mask",
            )
        require(
            item["token_truncated"] is False
            and np.asarray(item["tokenized_prompt"]).shape == (200,)
            and np.asarray(item["tokenized_prompt"]).dtype in (np.int32, np.int64)
            and np.asarray(item["tokenized_prompt_mask"]).shape == (200,)
            and np.asarray(item["tokenized_prompt_mask"]).dtype == np.bool_,
            "Missing/truncated FR3 tokenizer fixture",
        )

    def stack(key: str) -> torch.Tensor:
        return torch.from_numpy(np.stack([i[key] for i in items]))

    observation = Observation.from_dict(
        {
            "state": stack("state"),
            "image": {
                k: torch.from_numpy(np.stack([i["image"][k] for i in items]))
                for k in image_keys
            },
            "image_mask": {
                k: torch.from_numpy(np.stack([i["image_mask"][k] for i in items]))
                for k in image_keys
            },
            "tokenized_prompt": stack("tokenized_prompt").long(),
            "tokenized_prompt_mask": stack("tokenized_prompt_mask"),
        }
    )
    return {
        "observation": observation,
        "actions": stack("actions"),
        "native_state": stack("native_state"),
        "action_loss_mask": stack("action_loss_mask"),
        "valid_action_channels": stack("valid_action_channels"),
        "valid_future_steps": stack("valid_future_steps"),
        "training_schema": TRAINING_SCHEMA,
        "action_loss_mask_required": True,
        "offline_schema_probe": True,
        "physical_execution": False,
        "provenance": [copy.deepcopy(i["provenance"]) for i in items],
        "task_fixture": [copy.deepcopy(i["task"]) for i in items],
    }


def build_fr3_training_dataloader(
    dataset: FR3TrainingDataset, *, batch_size: int = 2
) -> StatefulDataLoader:
    """Build the native stateful loader for hardware-inaccessible schema probes."""
    require(
        isinstance(dataset, FR3TrainingDataset)
        and type(batch_size) is int
        and batch_size > 0,
        "Expected FR3 training dataset and positive batch size",
    )
    loader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_fr3_training_items,
    )
    loader.uses_action_loss_masks = True
    return loader
