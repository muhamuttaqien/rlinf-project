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

"""Typed episode/frame annotations for future RECAP use of FR3 archives."""

import math
import re
from dataclasses import dataclass
from typing import Literal

from rlinf.data.storage.fr3_validation import require

METADATA_SCHEMA = "fr3_recap_metadata_v1"
SourceType = Literal[
    "synthetic_demonstration",
    "human_demonstration",
    "policy_rollout",
    "correction",
    "offline_replay",
]
Split = Literal["unassigned", "train", "validation", "test"]


def _text(value: str) -> None:
    require(
        isinstance(value, str) and bool(value.strip()),
        "Expected nonempty annotation identity/text",
    )


def _digest(value: str) -> None:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "Expected SHA-256 annotation binding",
    )


@dataclass(frozen=True)
class AnnotationProvenance:
    """Identify who/what produced a label and its own timestamp domain."""

    source_id: str
    method: str
    version: str
    timestamp_ns: int
    clock_domain: str
    origin: Literal["synthetic_fixture", "human_annotation", "policy_evaluation"]

    def validate(self) -> None:
        """Validate provenance without comparing unrelated clock domains."""
        for value in (self.source_id, self.method, self.version, self.clock_domain):
            _text(value)
        require(
            type(self.timestamp_ns) is int and self.timestamp_ns > 0,
            "Expected positive annotation timestamp",
        )
        require(
            self.origin
            in ("synthetic_fixture", "human_annotation", "policy_evaluation"),
            "Unknown annotation origin",
        )


@dataclass(frozen=True)
class TaskIdentity:
    """Preserve original or explicitly synthetic instruction text and version."""

    task_id: str
    instruction: str
    version: str
    origin: Literal["recorded_original", "synthetic_fixture"]
    provenance: AnnotationProvenance

    def validate(self) -> None:
        """Reject empty or unattributed task instructions."""
        for value in (self.task_id, self.instruction, self.version):
            _text(value)
        require(
            self.origin in ("recorded_original", "synthetic_fixture"),
            "Unknown task instruction origin",
        )
        self.provenance.validate()
        if self.origin == "synthetic_fixture":
            require(
                self.provenance.origin == "synthetic_fixture",
                "Synthetic task needs synthetic annotation provenance",
            )


@dataclass(frozen=True)
class PolicyIdentity:
    """Identify a policy/configuration independently of native fixture weight UUIDs."""

    policy_id: str
    checkpoint_sha256: str
    configuration_sha256: str

    def validate(self) -> None:
        """Require explicit checkpoint and configuration digests."""
        _text(self.policy_id)
        _digest(self.checkpoint_sha256)
        _digest(self.configuration_sha256)


@dataclass(frozen=True)
class Intervention:
    """One operator interval, with an exclusive end frame."""

    start_frame: int
    end_frame: int
    operator_id: str
    reason: str
    provenance: AnnotationProvenance

    def validate(self, frame_count: int) -> None:
        """Check interval bounds and annotation ownership."""
        require(
            type(self.start_frame) is type(self.end_frame) is int
            and 0 <= self.start_frame < self.end_frame <= frame_count,
            "Intervention interval outside episode",
        )
        _text(self.operator_id)
        _text(self.reason)
        self.provenance.validate()


@dataclass(frozen=True)
class FrameAnnotation:
    """Optional evaluated labels, distinct from the unscored native fixture reward."""

    frame_index: int
    provenance: AnnotationProvenance
    reward: float | None = None
    return_value: float | None = None
    advantage_score: float | None = None
    advantage_label: bool | None = None

    def validate(self, frame_count: int) -> None:
        """Validate finite labels and frame association without calculating labels."""
        require(
            type(self.frame_index) is int and 0 <= self.frame_index < frame_count,
            "Annotation frame outside episode",
        )
        self.provenance.validate()
        for value in (self.reward, self.return_value, self.advantage_score):
            require(
                value is None or (type(value) in (int, float) and math.isfinite(value)),
                "Expected finite numeric RECAP labels or null",
            )
        require(
            self.advantage_label is None or type(self.advantage_label) is bool,
            "Expected Boolean advantage label or null",
        )
        require(
            any(
                x is not None
                for x in (
                    self.reward,
                    self.return_value,
                    self.advantage_score,
                    self.advantage_label,
                )
            ),
            "Empty frame annotation must be omitted",
        )


@dataclass(frozen=True)
class Outcome:
    """An attributed episode outcome; absence never means failure or success."""

    label: Literal["success", "failure", "unscored"]
    success: bool | None
    provenance: AnnotationProvenance

    def validate(self) -> None:
        """Require outcome labels and success flags to agree."""
        self.provenance.validate()
        require(
            self.label in ("success", "failure", "unscored"),
            "Unknown outcome annotation",
        )
        expected = {"success": True, "failure": False, "unscored": None}[self.label]
        require(
            self.success is expected,
            "Outcome success flag disagrees with annotation label",
        )


@dataclass(frozen=True)
class EpisodeMetadata:
    """Version-one offline annotation binding for one original native episode."""

    dataset_id: str
    source_manifest_sha256: str
    episode_id: str
    native_provenance_path: str
    source_type: SourceType = "synthetic_demonstration"
    split: Split = "unassigned"
    task: TaskIdentity | None = None
    policy: PolicyIdentity | None = None
    rollout_id: str | None = None
    iteration: int | None = None
    interventions: tuple[Intervention, ...] = ()
    frame_annotations: tuple[FrameAnnotation, ...] = ()
    outcome: Outcome | None = None
    physical_execution: Literal[False] = False
    controller_acceptance: Literal["UNAVAILABLE"] = "UNAVAILABLE"

    def validate(self, frame_count: int) -> None:
        """Check category requirements while preserving missing annotations."""
        for value in (self.dataset_id, self.episode_id, self.native_provenance_path):
            _text(value)
        _digest(self.source_manifest_sha256)
        require(
            self.split in ("unassigned", "train", "validation", "test"),
            "Unknown episode split",
        )
        require(
            self.source_type
            in (
                "synthetic_demonstration",
                "human_demonstration",
                "policy_rollout",
                "correction",
                "offline_replay",
            ),
            "Unknown dataset source type",
        )
        require(
            self.physical_execution is False
            and self.controller_acceptance == "UNAVAILABLE",
            "Physical execution/acceptance cannot be enabled by offline annotations",
        )
        if self.task:
            self.task.validate()
        if self.policy:
            self.policy.validate()
        if self.rollout_id is not None:
            _text(self.rollout_id)
            require(
                self.policy is not None
                and type(self.iteration) is int
                and self.iteration >= 0,
                "Rollout requires policy and nonnegative iteration",
            )
        else:
            require(
                self.iteration is None and self.policy is None,
                "Policy/iteration requires rollout identity",
            )
        if self.source_type == "policy_rollout":
            require(
                self.rollout_id is not None,
                "Policy rollout source requires rollout identity",
            )
        if self.source_type == "correction":
            require(
                bool(self.interventions),
                "Correction source requires intervention annotations",
            )
        for annotation in (*self.interventions, *self.frame_annotations):
            annotation.validate(frame_count)
        frames = [a.frame_index for a in self.frame_annotations]
        require(len(frames) == len(set(frames)), "Duplicate frame annotation")
        if self.outcome:
            self.outcome.validate()

    @classmethod
    def from_dict(cls, raw: dict) -> "EpisodeMetadata":
        """Parse typed nested annotations; unknown fields and malformed values fail."""
        try:
            require(
                isinstance(raw, dict) and set(raw) == set(cls.__dataclass_fields__),
                "Missing or unknown RECAP episode annotation fields",
            )
            values = dict(raw)
            for key, kind in (("task", TaskIdentity), ("outcome", Outcome)):
                if values.get(key) is not None:
                    entry = dict(values[key])
                    entry["provenance"] = AnnotationProvenance(**entry["provenance"])
                    values[key] = kind(**entry)
            if values.get("policy") is not None:
                values["policy"] = PolicyIdentity(**values["policy"])
            for key, kind in (
                ("interventions", Intervention),
                ("frame_annotations", FrameAnnotation),
            ):
                entries = []
                for entry in values.get(key, []):
                    entry = dict(entry)
                    entry["provenance"] = AnnotationProvenance(**entry["provenance"])
                    entries.append(kind(**entry))
                values[key] = tuple(entries)
            return cls(**values)
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid typed FR3 RECAP metadata: {exc}") from exc


def validate_episode_metadata(
    records: list[EpisodeMetadata],
    *,
    dataset_id: str,
    source_manifest_sha256: str,
    episodes: dict[str, tuple[int, str]],
) -> dict:
    """Bind synthetic annotations to native episodes and prevent split leakage."""
    seen = set()
    missing = {
        "task_instruction": 0,
        "split": 0,
        "policy_rollout": 0,
        "interventions": 0,
        "outcome": 0,
        "reward_return_advantage": 0,
    }
    for record in records:
        require(
            record.episode_id not in seen,
            "Episode split leakage or duplicate annotation identity",
        )
        seen.add(record.episode_id)
        require(
            record.dataset_id == dataset_id
            and record.source_manifest_sha256 == source_manifest_sha256,
            "RECAP dataset/manifest identity mismatch",
        )
        require(record.episode_id in episodes, "Unknown annotated episode")
        count, relative = episodes[record.episode_id]
        record.validate(count)
        require(
            record.native_provenance_path == relative,
            "Native provenance locator mismatch",
        )
        require(
            record.source_type in ("synthetic_demonstration", "offline_replay"),
            "Synthetic data misclassified as human demonstration, policy rollout or correction",
        )
        require(
            record.policy is None and record.rollout_id is None,
            "No policy rollout occurred in this synthetic source",
        )
        if record.task is not None:
            require(
                record.task.origin == "synthetic_fixture",
                "Original task instruction is missing from this native archive",
            )
        provenance = [
            a.provenance for a in (*record.interventions, *record.frame_annotations)
        ]
        if record.outcome:
            provenance.append(record.outcome.provenance)
        require(
            all(p.origin == "synthetic_fixture" for p in provenance),
            "Synthetic source annotations must be explicitly synthetic test data",
        )
        missing["task_instruction"] += int(
            record.task is None or record.task.origin != "recorded_original"
        )
        missing["split"] += int(record.split == "unassigned")
        missing["policy_rollout"] += int(record.rollout_id is None)
        missing["interventions"] += int(not record.interventions)
        missing["outcome"] += int(record.outcome is None)
        missing["reward_return_advantage"] += int(not record.frame_annotations)
    require(seen == set(episodes), "Missing RECAP episode annotation records")
    return {
        "schema": METADATA_SCHEMA,
        "status": "PASS_STRUCTURAL_ONLY",
        "episodes": len(records),
        "missing": missing,
        "experimental_readiness": "UNQUALIFIED_SYNTHETIC_ONLY",
    }
