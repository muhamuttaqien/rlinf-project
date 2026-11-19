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

"""Immutable, simulation-only action contracts and admission checks."""

import enum
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any

from .protocol import PROFILES

FR3_KIND = "fr3_normalized_pose_offset_v1"
ROBOTIQ_KIND = "robotiq_normalized_closure_v1"
LOCAL_CLOCK_DOMAIN = "local-monotonic:" + uuid.uuid4().hex
MAX_OBSERVATION_BYTES = 16384


class ActionState(str, enum.Enum):
    """Local/mock dispositions; none asserts physical execution."""

    CREATED = "created"
    VALIDATED = "validated"
    REJECTED = "rejected"
    ADMITTED = "admitted"
    SUBMITTED = "submitted"
    ACCEPTANCE_UNKNOWN = "acceptance_unknown"
    ACKNOWLEDGED = "mock_acknowledged"
    COMPLETED = "mock_completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELED = "mock_canceled"


class ActionRejected(ValueError):
    """An explicit admission failure with a stable machine-readable code."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class InterfaceIdentity:
    """Pinned mock controller/configuration and explicit endpoint semantics."""

    kind: str
    controller_id: str
    configuration_id: str
    endpoint: str
    units: str
    reference: str
    translation_frame: str | None = None
    rotation_frame: str | None = None
    controlled_link: str | None = None
    rotation_composition: str | None = None
    translation_scale_m: float | None = None
    rotation_scale_rad: float | None = None
    mounting_rpy_rad: tuple[float, ...] = ()

    @classmethod
    def mock(
        cls, kind: str, controller_id: str, configuration_id: str
    ) -> "InterfaceIdentity":
        """Describe one fixed offline endpoint; accept no ROS domain or live URI."""
        if kind == FR3_KIND:
            return cls(
                kind,
                controller_id,
                configuration_id,
                "mock://fr3/twist",
                "dimensionless",
                "measured_fk",
                "inverse_mount_into_base",
                "measured_fr3_link8",
                "fr3_link8",
                "right_multiplied_rpy",
                0.007,
                0.03,
                (0.0, 0.0, 0.0),
            )
        if kind == ROBOTIQ_KIND:
            return cls(
                kind,
                controller_id,
                configuration_id,
                "mock://robotiq/gripper",
                "normalized_closure_0_open_1_closed",
                "absolute_closure_request",
            )
        raise ActionRejected(
            "kind", f"Expected one of {FR3_KIND}, {ROBOTIQ_KIND}; got {kind!r}"
        )


@dataclass(frozen=True)
class ObservationProvenance:
    """An immutable copy of a 2A observation and its local clock identity.

    ROS source stamps remain inside the snapshot and are never subtracted from
    monotonic receipt times. This contains observations, not a hardware approval.
    """

    clock_domain: str
    snapshot_json: str
    timing_uncertainty_ns: int

    @classmethod
    def from_observation(
        cls,
        observation: dict[str, Any],
        *,
        clock_domain: str,
        timing_uncertainty_ns: int,
    ) -> "ObservationProvenance":
        """Copy an existing observation without opening or owning its connection."""
        return cls(
            clock_domain,
            json.dumps(observation, allow_nan=False, sort_keys=True),
            timing_uncertainty_ns,
        )


@dataclass(frozen=True)
class ActionRequest:
    """Requested normalized action; validation creates separate event records."""

    kind: str
    values: tuple[float, ...]
    source_id: str
    session_id: str
    authority_id: str
    sequence: int
    generated_ns: int
    deadline_ns: int
    clock_domain: str
    observation: ObservationProvenance
    interface: InterfaceIdentity
    schema_version: int = 1
    disposition: ActionState = ActionState.CREATED


@dataclass(frozen=True)
class MockLimitProfile:
    """Explicit synthetic limits/calibration approval, valid only for simulation.

    No default numeric limits or calibration approval are supplied. These values
    cannot authorize hardware execution or assert physical force/width control.
    """

    profile_id: str
    approval_id: str
    calibration_id: str
    interface: InterfaceIdentity
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    max_action_age_ns: int
    max_lifetime_ns: int
    max_observation_age_ns: int
    max_observation_skew_ns: int
    max_timing_uncertainty_ns: int
    acknowledgement_timeout_ns: int
    result_timeout_ns: int
    scope: str = "offline_synthetic"


@dataclass(frozen=True)
class MockAuthorization:
    """Source-bound simulation authorization for one connection's authority epoch."""

    authorization_id: str
    source_id: str
    session_id: str
    authority_id: str
    profile_id: str
    clock_domain: str
    deadline_ns: int
    scope: str = "offline_synthetic"


@dataclass
class ManualClock:
    """Deterministic local clock for offline tests; advance explicitly."""

    timestamp_ns: int = 1_000_000_000
    domain: str = field(default_factory=lambda: "manual:" + uuid.uuid4().hex)

    def __post_init__(self) -> None:
        integer(self.timestamp_ns, "clock_time", minimum=0)
        identifier(self.domain, "clock_domain")

    def now_ns(self) -> int:
        """Return the simulated monotonic time."""
        return self.timestamp_ns

    def advance(self, duration_ns: int) -> None:
        """Advance without allowing clock regression."""
        integer(duration_ns, "clock_advance", minimum=0)
        advanced = self.timestamp_ns + duration_ns
        integer(advanced, "clock_time", minimum=0)
        self.timestamp_ns = advanced


@dataclass(frozen=True)
class MonotonicClock:
    """Current-process monotonic clock; identity is not transferable across hosts."""

    domain: str = LOCAL_CLOCK_DOMAIN

    def now_ns(self) -> int:
        """Return host monotonic nanoseconds."""
        return time.monotonic_ns()


def reject(condition: bool, code: str, detail: str) -> None:
    """Reject a failed contract predicate."""
    if not condition:
        raise ActionRejected(code, detail)


def integer(value: Any, code: str, *, minimum: int = 1) -> None:
    """Require an integer, excluding booleans and below-bound values."""
    reject(
        type(value) is int and minimum <= value <= (1 << 63) - 1,
        code,
        f"Expected a signed-64-bit integer >= {minimum}; got {type(value).__name__}",
    )


def identifier(value: Any, code: str) -> None:
    """Require a bounded nonempty identifier."""
    reject(
        isinstance(value, str) and bool(value.strip()) and len(value) <= 128,
        code,
        "Expected a nonempty identifier of at most 128 characters",
    )


def vector(value: Any, width: int, code: str) -> None:
    """Require an exact immutable vector of finite real numbers."""
    reject(
        isinstance(value, tuple) and len(value) == width,
        code,
        f"Expected a tuple with exactly {width} values",
    )
    try:
        valid = all(
            isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v)
            for v in value
        )
    except (OverflowError, TypeError, ValueError):
        valid = False
    reject(valid, code, "Expected finite real numbers, excluding booleans")


def validate_approval(
    profile: MockLimitProfile,
    authorization: MockAuthorization,
    interface: InterfaceIdentity,
    session_id: str,
    authority_id: str,
    clock_domain: str,
    now_ns: int,
) -> None:
    """Validate source/session-bound approval without permitting physical scope."""
    reject(
        type(profile) is MockLimitProfile,
        "profile",
        "Expected explicit synthetic limit profile",
    )
    reject(
        type(authorization) is MockAuthorization,
        "authorization",
        "Expected explicit mock authorization",
    )
    reject(
        profile.scope == authorization.scope == "offline_synthetic",
        "scope",
        "Only offline_synthetic approval is supported",
    )
    for value in (
        profile.profile_id,
        profile.approval_id,
        profile.calibration_id,
        authorization.authorization_id,
        authorization.source_id,
        interface.controller_id,
        interface.configuration_id,
    ):
        identifier(value, "approval_identity")
    reject(
        interface
        == InterfaceIdentity.mock(
            interface.kind, interface.controller_id, interface.configuration_id
        ),
        "interface",
        "Expected the fixed mock interface semantics",
    )
    reject(
        profile.interface == interface,
        "interface",
        "Profile/controller/configuration mismatch",
    )
    reject(
        (
            authorization.session_id,
            authorization.authority_id,
            authorization.clock_domain,
            authorization.profile_id,
        )
        == (session_id, authority_id, clock_domain, profile.profile_id),
        "authorization",
        "Authorization must match this session, authority, clock and profile",
    )
    integer(authorization.deadline_ns, "authorization_deadline")
    reject(
        now_ns < authorization.deadline_ns,
        "authorization_expired",
        "Mock authorization has expired",
    )
    width = 6 if interface.kind == FR3_KIND else 1
    vector(profile.lower, width, "profile_limits")
    vector(profile.upper, width, "profile_limits")
    floor = -1 if interface.kind == FR3_KIND else 0
    reject(
        all(
            floor <= low < high <= 1 for low, high in zip(profile.lower, profile.upper)
        ),
        "profile_limits",
        "Expected ordered synthetic limits inside the representation range",
    )
    for name in (
        "max_action_age_ns",
        "max_lifetime_ns",
        "max_observation_age_ns",
        "acknowledgement_timeout_ns",
        "result_timeout_ns",
    ):
        integer(getattr(profile, name), "profile_timing")
    for name in ("max_observation_skew_ns", "max_timing_uncertainty_ns"):
        integer(getattr(profile, name), "profile_timing", minimum=0)


def validate_observation(
    provenance: ObservationProvenance,
    kind: str,
    profile: MockLimitProfile,
    clock_domain: str,
    now_ns: int,
) -> None:
    """Check 2A snapshot identity, stream completeness, receipt ages and skew."""
    reject(
        type(provenance) is ObservationProvenance,
        "observation",
        "Expected observation provenance",
    )
    reject(
        provenance.clock_domain == clock_domain,
        "clock_domain",
        "Observation belongs to another monotonic clock domain",
    )
    integer(provenance.timing_uncertainty_ns, "observation_uncertainty", minimum=0)
    reject(
        provenance.timing_uncertainty_ns <= profile.max_timing_uncertainty_ns,
        "observation_uncertainty",
        "Observation timing uncertainty exceeds the synthetic profile",
    )
    reject(
        isinstance(provenance.snapshot_json, str)
        and len(provenance.snapshot_json.encode()) <= MAX_OBSERVATION_BYTES,
        "observation",
        "Expected a bounded JSON observation snapshot",
    )
    try:
        snapshot = json.loads(provenance.snapshot_json)
        # Re-encoding rejects NaN/Infinity even if a caller constructed the JSON directly.
        json.dumps(snapshot, allow_nan=False)
        status = snapshot["ros2"]
        reject(
            status["complete"] is True
            and status["bridge_alive"] is True
            and status.get("error") is None,
            "observation",
            "Expected a complete healthy observation snapshot",
        )
        reject(
            all(
                status.get(key, []) == []
                for key in (
                    "missing_streams",
                    "stale_streams",
                    "invalid_streams",
                    "missing_fields",
                )
            ),
            "observation",
            "Observation completeness flags are inconsistent",
        )
        arm = kind == FR3_KIND
        reject(
            type(status["domain_id"]) is int
            and status["domain_id"] == (70 if arm else 71),
            "observation_interface",
            "Unexpected observation source domain metadata",
        )
        specs = PROFILES["fr3" if arm else "robotiq"]
        streams = status["streams"]
        reject(
            set(streams) == set(specs),
            "observation",
            "Expected every selected 2A stream",
        )
        receipts = []
        for key, (topic, ros_type, _) in specs.items():
            stream = streams[key]
            reject(
                (stream["topic"], stream["type"]) == (topic, ros_type),
                "observation_interface",
                "Observation topic/type mismatch",
            )
            reject(
                stream["connected"] is True
                and stream["fresh"] is True
                and not stream.get("error")
                and not stream.get("graph_error"),
                "observation",
                "Observation stream is disconnected or invalid",
            )
            integer(stream["sequence"], "observation_sequence")
            sample = stream["sample"]
            stamp = sample["header"]
            reject(
                type(stamp["sec"]) is int
                and type(stamp["nanosec"]) is int
                and 0 <= stamp["nanosec"] < 1_000_000_000
                and isinstance(stamp["frame_id"], str),
                "observation_header",
                "Expected a ROS header with integer source stamp and frame",
            )
            integer(sample["received_monotonic_ns"], "observation_receipt", minimum=0)
            receipt = sample["received_monotonic_ns"]
            reject(
                0 <= now_ns - receipt <= profile.max_observation_age_ns,
                "observation_stale",
                "Observation receipt is future-dated or stale",
            )
            reject(
                isinstance(sample["data"], dict) and bool(sample["data"]),
                "observation",
                "Expected decoded measured data",
            )
            receipts.append(receipt)
        reject(
            max(receipts) - min(receipts) <= profile.max_observation_skew_ns,
            "observation_skew",
            "Observation receipt skew exceeds the synthetic profile",
        )
        reject(
            isinstance(snapshot["state"], dict)
            and bool(snapshot["state"])
            and all(v is not None for v in snapshot["state"].values()),
            "observation",
            "Expected present selected measured state fields",
        )
    except (KeyError, TypeError, ValueError, AttributeError, RecursionError) as exc:
        if isinstance(exc, ActionRejected):
            raise
        raise ActionRejected(
            "observation", "Malformed 2A observation snapshot"
        ) from exc


def validate_request(
    request: ActionRequest,
    profile: MockLimitProfile,
    authorization: MockAuthorization,
    interface: InterfaceIdentity,
    session_id: str,
    authority_id: str,
    clock_domain: str,
    last_sequence: int,
    now_ns: int,
) -> None:
    """Validate executable admission to the local mock only; never clip values."""
    validate_approval(
        profile,
        authorization,
        interface,
        session_id,
        authority_id,
        clock_domain,
        now_ns,
    )
    reject(type(request) is ActionRequest, "schema", "Expected an ActionRequest")
    reject(
        type(request.schema_version) is int
        and request.schema_version == 1
        and request.disposition is ActionState.CREATED,
        "schema",
        "Expected schema 1 with created disposition",
    )
    reject(
        request.kind == interface.kind, "kind", "Action kind does not match this part"
    )
    reject(
        request.interface == interface,
        "interface",
        "Action/controller/configuration semantics mismatch",
    )
    reject(
        request.clock_domain == clock_domain,
        "clock_domain",
        "Action belongs to another monotonic clock domain",
    )
    reject(
        (request.source_id, request.session_id, request.authority_id)
        == (authorization.source_id, session_id, authority_id),
        "authority",
        "Action source/session/authority mismatch",
    )
    identifier(request.source_id, "source")
    integer(request.sequence, "sequence")
    reject(
        request.sequence > last_sequence,
        "replay",
        "Sequence must exceed the last admitted sequence",
    )
    integer(request.generated_ns, "timestamp", minimum=0)
    integer(request.deadline_ns, "deadline")
    reject(
        0 <= now_ns - request.generated_ns <= profile.max_action_age_ns,
        "action_age",
        "Action generation time is future-dated or stale",
    )
    reject(
        request.generated_ns < request.deadline_ns
        and now_ns < request.deadline_ns
        and request.deadline_ns <= authorization.deadline_ns
        and request.deadline_ns - request.generated_ns <= profile.max_lifetime_ns,
        "expiry",
        "Action is expired or exceeds the permitted lifetime",
    )
    vector(request.values, 6 if request.kind == FR3_KIND else 1, "payload")
    reject(
        all(
            low <= v <= high
            for v, low, high in zip(request.values, profile.lower, profile.upper)
        ),
        "range",
        "Action exceeds explicitly approved synthetic channel limits",
    )
    validate_observation(
        request.observation, request.kind, profile, clock_domain, now_ns
    )


def request_json(request: ActionRequest) -> str:
    """Serialize the immutable requested action and complete observation provenance."""
    return encode_json(asdict(request))


def admission_deadline_ns(
    request: ActionRequest, profile: MockLimitProfile, authorization: MockAuthorization
) -> int:
    """Bound submission by every validated source, observation and approval age."""
    streams = json.loads(request.observation.snapshot_json)["ros2"]["streams"]
    oldest = min(
        stream["sample"]["received_monotonic_ns"] for stream in streams.values()
    )
    return min(
        request.deadline_ns,
        authorization.deadline_ns,
        request.generated_ns + profile.max_action_age_ns + 1,
        oldest + profile.max_observation_age_ns + 1,
    )


def encode_json(value: Any) -> str:
    """Serialize finite real scalars, including numeric-array scalar types."""

    def number(item: Any) -> float:
        if isinstance(item, Real) and not isinstance(item, bool):
            return float(item)
        raise TypeError(
            f"Expected a JSON value or real scalar; got {type(item).__name__}"
        )

    return json.dumps(value, default=number, allow_nan=False, sort_keys=True)
