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

"""Read-only native LeRobot loading for the FR3 structural OpenPI contract."""

import bisect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from rlinf.data.datasets.openpi.fr3_contract import (
    CONTRACT_PATH,
    load_fr3_contract,
    verify_openpi_installation,
)
from rlinf.data.datasets.recap.fr3_metadata import (
    METADATA_SCHEMA,
    EpisodeMetadata,
    validate_episode_metadata,
)
from rlinf.data.storage.fr3_synthetic import read_json
from rlinf.data.storage.fr3_validation import artifact, load_records, require
from rlinf.data.storage.lerobot.compat import episode_boundaries
from rlinf.data.storage.lerobot.fr3_export import REPO_ID, audit_synthetic_export
from rlinf.models.embodiment.openpi.policies.fr3_policy import FR3Inputs


@dataclass(frozen=True)
class FR3DatasetConfig:
    """Structural configuration; deliberately not a trainable OpenPI TrainConfig."""

    contract_path: str | Path = CONTRACT_PATH
    action_horizon: int = 4
    padding_policy: str = "lerobot_repeat_last_with_mask_v1"
    split: str | None = None

    def validate(self) -> dict:
        """Resolve only reviewed contract and window policies."""
        contract = load_fr3_contract(self.contract_path)
        require(
            type(self.action_horizon) is int and 1 <= self.action_horizon <= 1000,
            "Expected action horizon integer in [1,1000]",
        )
        require(
            self.padding_policy
            in (
                contract["temporal"]["padding_policy"],
                contract["temporal"]["strict_policy"],
            ),
            "Unsupported action-window padding policy",
        )
        require(
            self.split is None or self.split in contract["recap"]["splits"],
            "Unknown requested episode split",
        )
        return contract


class FR3OfflineDataset(Dataset):
    """Load an audited export bundle while retaining its native provenance archive.

    Samples contain structural observations and recorded controller requests.
    They contain no model training target, prompt substitution or executable action.
    """

    def __init__(
        self,
        export_path: str | Path,
        *,
        config: FR3DatasetConfig | None = None,
        annotations_path: str | Path | None = None,
    ) -> None:
        self.config = config or FR3DatasetConfig()
        self.contract = self.config.validate()
        self.runtime = verify_openpi_installation(self.contract)
        self.root = Path(export_path).absolute()
        self.audit = audit_synthetic_export(self.root)
        export = read_json(artifact(self.root, "export.json"))
        self.native_root = self.root / "native"
        self.manifest = read_json(artifact(self.native_root, "manifest.json"))
        self.source_manifest_sha256 = export["source_manifest_sha256"]
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        period = self.manifest["step_period_ns"]
        require(
            1_000_000_000 % period == 0, "FR3 loader cannot resample non-integer FPS"
        )
        self._dataset = LeRobotDataset(
            REPO_ID,
            root=self.root / "lerobot",
            delta_timestamps={
                "actions": [t * period / 1e9 for t in range(self.config.action_horizon)]
            },
            download_videos=False,
        )
        require(
            self._dataset.fps * period == 1_000_000_000,
            "LeRobot and native timing frequency mismatch",
        )
        self._starts, self._ends = episode_boundaries(self._dataset)
        self._episodes, self._records = [], []
        bindings = {}
        for number, entry in enumerate(self.manifest["episodes"]):
            episode = read_json(artifact(self.native_root, entry["metadata_path"]))
            self._episodes.append(episode)
            self._records.append(
                load_records(artifact(self.native_root, entry["provenance_path"]))
            )
            bindings[episode["episode_id"]] = (
                episode["step_count"],
                entry["provenance_path"],
            )
            require(
                self._ends[number] - self._starts[number] == episode["step_count"],
                "Native/export episode boundary mismatch",
            )
        require(
            self._starts[0] == 0
            and self._ends[-1] == len(self._dataset)
            and self._starts[1:] == self._ends[:-1],
            "Noncontiguous LeRobot episode index",
        )
        if annotations_path is None:
            metadata = [
                EpisodeMetadata(
                    dataset_id=self.manifest["dataset_id"],
                    source_manifest_sha256=self.source_manifest_sha256,
                    episode_id=name,
                    native_provenance_path=relative,
                )
                for name, (_, relative) in bindings.items()
            ]
        else:
            raw = read_json(Path(annotations_path))
            require(
                isinstance(raw, dict)
                and set(raw) == {"schema", "episodes"}
                and raw["schema"] == METADATA_SCHEMA
                and isinstance(raw["episodes"], list),
                "Unknown/missing RECAP metadata schema",
            )
            metadata = [EpisodeMetadata.from_dict(r) for r in raw["episodes"]]
        self.metadata_report = validate_episode_metadata(
            metadata,
            dataset_id=self.manifest["dataset_id"],
            source_manifest_sha256=self.source_manifest_sha256,
            episodes=bindings,
        )
        self._metadata = {r.episode_id: r for r in metadata}
        self._indices = [
            i
            for number, episode in enumerate(self._episodes)
            if self.config.split is None
            or self._metadata[episode["episode_id"]].split == self.config.split
            for i in range(self._starts[number], self._ends[number])
        ]
        require(
            bool(self._indices),
            "No episodes assigned to requested split; unassigned data cannot become training data",
        )
        self._transform = FR3Inputs(contract_path=self.config.contract_path)

    def __len__(self) -> int:
        """Return the number of original transitions, including short windows."""
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an episode-bounded structural sample with explicit padding provenance."""
        require(
            type(index) is int and 0 <= index < len(self),
            "FR3 dataset index outside original transitions",
        )
        index = self._indices[index]
        number = bisect.bisect_right(self._starts, index) - 1
        start, end = self._starts[number], self._ends[number]
        step, episode = index - start, self._episodes[number]
        item = self._dataset[index]
        require(
            int(item["episode_index"]) == number and int(item["frame_index"]) == step,
            "LeRobot episode/frame association mismatch",
        )
        horizon = self.config.action_horizon
        expected_indices = np.minimum(np.arange(step, step + horizon), end - start - 1)
        expected_padding = np.arange(step, step + horizon) >= end - start
        padding = item["actions_is_pad"].numpy()
        require(
            padding.dtype == np.bool_ and np.array_equal(padding, expected_padding),
            "Incorrect native action-window padding mask",
        )
        if self.config.padding_policy == "require_full_window_v1":
            require(
                not padding.any(),
                "Missing future actions: strict window would require episode padding",
            )
        records = self._records[number]
        timestamps, expected_actions = [], []
        for source_step in expected_indices:
            request = records[int(source_step) * 2]
            timestamps.append(request["logical_action_ns"])
            expected_actions.append(
                request["typed_requests"]["arm"]["values"]
                + request["typed_requests"]["end_effector"]["values"]
            )
        actions = item["actions"].numpy()
        require(
            actions.shape == (horizon, 7)
            and np.array_equal(actions, np.array(expected_actions, dtype=np.float32)),
            "Action window differs from native episode requests",
        )
        valid_times = np.asarray(timestamps)[~padding]
        require(
            int(item["logical_ns"]) == timestamps[0]
            and (np.diff(valid_times) == self.manifest["step_period_ns"]).all(),
            "Inconsistent temporal action-window timestamps",
        )
        request, outcome = records[2 * step : 2 * step + 2]
        observation = request["observation_before"]
        context = {
            "synthetic": True,
            "physical_execution": False,
            "dataset_id": self.manifest["dataset_id"],
            "source_manifest_sha256": self.source_manifest_sha256,
            "episode_id": episode["episode_id"],
            "episode_index": number,
            "frame_index": step,
            "source_frame_indices": expected_indices.tolist(),
            "source_timestamps_ns": timestamps,
            "clock_domain": self.manifest["synthetic_clock_domain"],
            "padding_policy": self.config.padding_policy,
            "state_fields": self.manifest["state_fields"],
            "joint_names": observation["arm"]["ros2"]["streams"]["joints"]["sample"][
                "data"
            ]["names"],
            "frame_contract": {
                f["name"]: f["frame"] for f in self.contract["state"]["fields"]
            },
            "camera_roles": self.manifest["camera_roles"],
            "action_kinds": self.manifest["action_kinds"],
            "action_semantics": self.contract["actions"]["semantics"],
            "metadata": asdict(self._metadata[episode["episode_id"]]),
            "native_provenance_path": self.manifest["episodes"][number][
                "provenance_path"
            ],
            "operator_request": request["operator_request"],
            "typed_requests": request["typed_requests"],
            "mock_outcome": outcome,
            "observation": observation,
        }
        data = {
            "state": item["state"].numpy(),
            "actions": actions,
            "actions_is_pad": padding,
            "fr3_context": context,
        }
        for name, key in self.contract["cameras"]["native_slots"].items():
            image = item[key].numpy()
            require(
                image.ndim == 3
                and image.shape[0] == 3
                and np.isfinite(image).all()
                and image.min() >= 0
                and image.max() <= 1,
                f"Invalid decoded LeRobot camera: {name}",
            )
            data[key] = np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
        return self._transform(data)

    def metadata_document(self) -> dict:
        """Return a detached metadata document; missing labels remain explicit."""
        return {
            "schema": METADATA_SCHEMA,
            "episodes": [asdict(r) for r in self._metadata.values()],
        }

    def require_model_ready(self) -> None:
        """Reject training/inference under the structural-only contract."""
        self._transform.require_model_ready()
