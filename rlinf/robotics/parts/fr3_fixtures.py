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

"""Deterministic device samples in a shared synthetic clock domain."""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

STREAMS = ("arm", "gripper", "d405", "d435i", "spacemouse")
SOURCE_OFFSETS_NS = dict(
    zip(STREAMS, (-3_000_000, -2_000_000, -8_000_000, -5_000_000, -1_000_000))
)
ARRIVAL_OFFSETS_NS = dict(
    zip(STREAMS, (-2_000_000, -1_000_000, -4_000_000, -2_000_000, 0))
)
SYNTHETIC_EPOCH_NS = 1_700_000_000_000_000_000


@dataclass(frozen=True)
class SyntheticFrame:
    """One logical observation boundary, independent of mock action outcomes.

    Source and arrival offsets model asynchronous samples on one synthetic clock.
    Actual host receipt timestamps are recorded separately by the native parts.
    """

    index: int
    timestamp_ns: int
    clock_domain: str
    seed: int = 0
    source_offsets_ns: dict[str, int] = field(
        default_factory=lambda: dict(SOURCE_OFFSETS_NS)
    )
    arrival_offsets_ns: dict[str, int] = field(
        default_factory=lambda: dict(ARRIVAL_OFFSETS_NS)
    )
    missing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.index) is not int
            or self.index < 0
            or type(self.timestamp_ns) is not int
            or self.timestamp_ns < 10_000_000
        ):
            raise ValueError(
                "Expected nonnegative fixture index and positive synthetic time"
            )
        if (
            not self.clock_domain.startswith("synthetic:")
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError(
                "Expected a synthetic clock domain and nonnegative fixture seed"
            )
        for offsets in (self.source_offsets_ns, self.arrival_offsets_ns):
            if set(offsets) != set(STREAMS) or any(
                type(v) is not int for v in offsets.values()
            ):
                raise ValueError(
                    "Expected integer offsets for all five fixture streams"
                )
        if set(self.missing) - set(STREAMS):
            raise ValueError("Unexpected missing fixture stream")

    def timing(self, stream: str) -> dict[str, Any]:
        """Return modeled source/arrival times, never host-clock measurements."""
        return {
            "synthetic": True,
            "frame_index": self.index,
            "clock_domain": self.clock_domain,
            "snapshot_ns": self.timestamp_ns,
            "source_ns": self.timestamp_ns + self.source_offsets_ns[stream],
            "arrival_ns": self.timestamp_ns + self.arrival_offsets_ns[stream],
            "missing": stream in self.missing,
            "substituted": False,
        }

    def axes(self) -> tuple[float, ...]:
        """Generate raw scripted input; no physical device axis mapping is implied."""
        return tuple(
            0.2 * math.sin(self.index * 0.3 + i + self.seed * 0.01) for i in range(6)
        )

    def joint_positions(self) -> list[float]:
        """Generate fixture state independently of requested action values."""
        return [
            0.1 * math.sin(self.index * 0.1 + i + self.seed * 0.01) for i in range(7)
        ]

    def closure(self) -> float:
        """Generate a normalized request, without a physical width conversion."""
        return 0.5 + 0.3 * math.sin(self.index * 0.2 + self.seed * 0.01)

    def image(self, sensor: str, width: int, height: int) -> np.ndarray | None:
        """Generate deterministic RGB pixels or explicitly mark a dropped frame."""
        if sensor in self.missing:
            return None
        y, x, channel = np.indices((height, width, 3))
        return (
            (
                x
                + 3 * y
                + 17 * channel
                + self.seed
                + self.index * 7
                + (23 if sensor == "d405" else 0)
            )
            % 256
        ).astype(np.uint8)


def validate_alignment(
    timings: dict[str, dict[str, Any]], max_age_ns: int, max_skew_ns: int
) -> None:
    """Require a complete latest-before-action sample set on one synthetic clock."""
    if set(timings) != set(STREAMS):
        raise ValueError("Expected all five synthetic streams for alignment")
    domains = {v["clock_domain"] for v in timings.values()}
    snapshots = {v["snapshot_ns"] for v in timings.values()}
    indices = {v["frame_index"] for v in timings.values()}
    if (
        len(domains) != 1
        or not next(iter(domains)).startswith("synthetic:")
        or len(snapshots) != 1
        or len(indices) != 1
    ):
        raise ValueError("Expected one shared synthetic clock and observation boundary")
    for name, sample in timings.items():
        if sample["missing"] or sample["substituted"]:
            raise ValueError(f"missing_sample:{name}: missing or substituted fixture")
        if (
            not 0
            <= sample["source_ns"]
            <= sample["arrival_ns"]
            <= sample["snapshot_ns"]
        ):
            raise ValueError(
                f"delayed_sample:{name}: expected source <= arrival <= snapshot"
            )
        if sample["snapshot_ns"] - sample["source_ns"] > max_age_ns:
            raise ValueError(
                f"stale_sample:{name}: synthetic acquisition age exceeds fixture limit"
            )
    sources = [s["source_ns"] for s in timings.values()]
    if max(sources) - min(sources) > max_skew_ns:
        raise ValueError("sample_skew: synthetic source skew exceeds fixture limit")
