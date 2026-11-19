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

"""Native normalized action parts that can submit only to local fake endpoints."""

import json
import threading
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from .base import Action, ControllablePart, Features, Observation
from .transports.ros2.action_contract import (
    FR3_KIND,
    ROBOTIQ_KIND,
    ActionRejected,
    ActionRequest,
    ActionState,
    InterfaceIdentity,
    ManualClock,
    MockAuthorization,
    MockLimitProfile,
    MonotonicClock,
    admission_deadline_ns,
    encode_json,
    integer,
    reject,
    request_json,
    validate_approval,
    validate_request,
)
from .transports.ros2.mock_action import MockROS2ActionTransport


@dataclass
class _ActionRecord:
    request: ActionRequest
    state: ActionState = ActionState.ADMITTED
    submitted_ns: int | None = None
    acknowledged_ns: int | None = None


class OfflineROS2ActionPart(ControllablePart):
    """A disarmed-by-default mock action connection, with no live backend option.

    Args:
        controller_id: Pinned synthetic controller identity.
        configuration_id: Pinned synthetic configuration identity.
        clock: Local monotonic clock, or an explicitly advanced offline clock.
        event_capacity: Maximum retained events (1–4096); eviction is counted.
        ipc_timeout: Wall-clock timeout for local fake-worker IPC.
    """

    KIND: ClassVar[str]
    ACTION_FIELD: ClassVar[str]

    def __init__(
        self,
        *,
        controller_id: str = "mock:controller-v1",
        configuration_id: str = "synthetic:configuration-v1",
        clock: MonotonicClock | ManualClock | None = None,
        event_capacity: int = 128,
        ipc_timeout: float = 1.0,
    ) -> None:
        integer(event_capacity, "event_capacity")
        reject(
            event_capacity <= 4096, "event_capacity", "Expected event capacity <= 4096"
        )
        self.interface = InterfaceIdentity.mock(
            self.KIND, controller_id, configuration_id
        )
        self.clock = clock if clock is not None else MonotonicClock()
        self._transport = MockROS2ActionTransport(ipc_timeout)
        self._lock = threading.RLock()
        self._events: deque[dict[str, Any]] = deque(maxlen=event_capacity)
        self._event_count = 0
        self.session_id = ""
        self.authority_id = ""
        self._profile: MockLimitProfile | None = None
        self._authorization: MockAuthorization | None = None
        self._last_sequence = 0
        self._pending: _ActionRecord | None = None
        self._active: _ActionRecord | None = None
        self._last: _ActionRecord | None = None

    @property
    def armed(self) -> bool:
        """Whether this session currently has an explicit mock-only approval."""
        return self._authorization is not None

    def _open(self) -> MockROS2ActionTransport:
        with self._lock:
            self.session_id, self.authority_id = uuid.uuid4().hex, uuid.uuid4().hex
            self._profile = self._authorization = None
            self._pending = self._active = self._last = None
            self._last_sequence = 0
            self._transport.start(self.interface, self.session_id, self.authority_id)
            return self._transport

    def _release(self, device: MockROS2ActionTransport) -> None:
        with self._lock:
            try:
                self._disarm("disconnect")
            finally:
                device.close()

    def reset(self) -> None:
        """Disarm and cancel only simulated pending/outstanding requests."""
        with self._lock:
            self._disarm("reset")

    @property
    def action_features(self) -> Features:
        """Declare the distinct normalized request key; never pose, joint or width."""
        return {
            self.ACTION_FIELD: {
                "kind": self.KIND,
                "shape": (6 if self.KIND == FR3_KIND else 1,),
            }
        }

    @property
    def observation_features(self) -> Features:
        """Expose local dispositions and the associated immutable measured snapshot."""
        return {"mock_action": {}, "measured": {}}

    def events(self) -> list[dict[str, Any]]:
        """Return copied schema-1 event records; all explicitly deny physical execution."""
        with self._lock:
            return json.loads(json.dumps(list(self._events), allow_nan=False))

    def _event(
        self,
        request: Any,
        state: ActionState,
        reason: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        payload = None
        if type(request) is ActionRequest:
            try:
                serialized = request_json(request)
                if len(serialized.encode()) <= 32768:
                    payload = json.loads(serialized)
            except (TypeError, ValueError, OverflowError, RecursionError):
                pass
        self._event_count += 1
        try:
            preview = None if payload is not None else repr(request)[:1024]
        except (ValueError, TypeError, RecursionError):
            preview = f"<unrepresentable {type(request).__name__}>"
        self._events.append(
            {
                "schema_version": 1,
                "event_index": self._event_count,
                "timestamp_ns": self.clock.now_ns(),
                "clock_domain": self.clock.domain,
                "offline": True,
                "physical_execution": False,
                "state": state.value,
                "reason": reason,
                "request": payload,
                "requested_preview": preview,
                "detail": detail,
                "approval": None
                if not self.armed
                else json.loads(
                    encode_json(
                        {
                            "profile": asdict(self._profile),
                            "authorization": asdict(self._authorization),
                        }
                    )
                ),
            }
        )

    def _transition(
        self,
        record: _ActionRecord,
        state: ActionState,
        reason: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        record.state = state
        self._last = record
        self._event(record.request, state, reason, detail)

    def _disarm(
        self,
        reason: str,
        state: ActionState = ActionState.CANCELED,
        contact_worker: bool = True,
    ) -> None:
        if self._pending is not None:
            self._transition(self._pending, state, reason)
            self._pending = None
        if self._active is not None:
            if contact_worker:
                try:
                    self._transport.exchange(
                        "cancel",
                        now_ns=self.clock.now_ns(),
                        sequence=self._active.request.sequence,
                    )
                except ConnectionError:
                    state, reason = ActionState.FAILED, "cancel_ipc_failure"
            self._transition(self._active, state, reason)
            self._active = None
        self._profile = self._authorization = None

    def _rpc(self, op: str, **payload: Any) -> dict[str, Any]:
        try:
            return self._transport.exchange(op, now_ns=self.clock.now_ns(), **payload)
        except ConnectionError as exc:
            self._disarm("ipc_failure", ActionState.FAILED, contact_worker=False)
            raise ActionRejected("ipc_failure", str(exc)) from exc

    def _readiness(self) -> None:
        status = self._rpc("status")
        reject(
            status.get("consistent") is True,
            "readiness",
            "Mock controller status is inconsistent",
        )
        reject(
            status.get("owner_conflict") is False,
            "ownership",
            "Competing mock command authority",
        )
        reject(
            status.get("interface") == json.loads(json.dumps(self.interface.__dict__)),
            "interface",
            "Mock controller identity/configuration changed",
        )

    def arm_mock(
        self, profile: MockLimitProfile, authorization: MockAuthorization
    ) -> None:
        """Authorize simulation only, using an explicit synthetic limit/calibration profile."""
        with self._lock:
            try:
                reject(
                    self.is_connected and self._transport.is_alive,
                    "connection",
                    "Connect the offline part before authorization",
                )
                reject(
                    not self.armed,
                    "authorization",
                    "Reset/disarm before replacing mock authorization",
                )
                validate_approval(
                    profile,
                    authorization,
                    self.interface,
                    self.session_id,
                    self.authority_id,
                    self.clock.domain,
                    self.clock.now_ns(),
                )
                self._readiness()
                self._profile, self._authorization = profile, authorization
            except ActionRejected as exc:
                self._event(None, ActionState.REJECTED, exc.code)
                self._disarm(exc.code)
                raise

    def configure_mock_behavior(self, behavior: str, *, delay_ns: int = 0) -> None:
        """Select a deterministic failure/result scenario on the fake worker only."""
        with self._lock:
            integer(delay_ns, "mock_delay", minimum=0)
            self._rpc("behavior", behavior=behavior, delay_ns=delay_ns)

    def _validate(self, request: ActionRequest, *, pending: bool = False) -> None:
        reject(
            self.is_connected and self._transport.is_alive,
            "connection",
            "Expected a connected offline worker",
        )
        reject(self.armed, "authorization", "Explicit mock authorization is required")
        self._readiness()
        validate_request(
            request,
            self._profile,
            self._authorization,
            self.interface,
            self.session_id,
            self.authority_id,
            self.clock.domain,
            request.sequence - 1 if pending else self._last_sequence,
            self.clock.now_ns(),
        )
        reject(
            len(request_json(request).encode()) <= 32768,
            "request_size",
            "Expected request <= 32 KiB",
        )

    def queue_action(self, request: ActionRequest) -> Observation:
        """Admit to a single pending slot, replacing an older pending mock request.

        Admission consumes its sequence even if submission later fails. This call
        never submits; submit_pending() revalidates all time/identity conditions.
        """
        with self._lock:
            self._event(request, ActionState.CREATED)
            try:
                self._validate(request)
            except ActionRejected as exc:
                self._event(request, ActionState.REJECTED, exc.code)
                self._disarm(exc.code)
                raise
            self._event(request, ActionState.VALIDATED)
            self._last_sequence = request.sequence
            if self._pending is not None:
                self._transition(
                    self._pending, ActionState.CANCELED, "replaced_pending"
                )
            self._pending = _ActionRecord(request)
            self._transition(self._pending, ActionState.ADMITTED)
            return self._snapshot()

    def submit_pending(self) -> Observation:
        """Revalidate and submit one pending request to the mock subprocess."""
        with self._lock:
            reject(
                self._pending is not None,
                "pending",
                "Expected an admitted pending mock action",
            )
            record = self._pending
            try:
                self._validate(record.request, pending=True)
                valid_until = admission_deadline_ns(
                    record.request, self._profile, self._authorization
                )
                response = self._rpc(
                    "submit",
                    action=json.loads(request_json(record.request)),
                    admission_deadline_ns=valid_until,
                )
                reject(
                    response.get("state")
                    in ("acceptance_unknown", "mock_acknowledged"),
                    response.get("reason", "mock_submission_rejected"),
                    "Mock endpoint rejected submission",
                )
                reject(
                    response.get("sequence") == record.request.sequence
                    and isinstance(response.get("wire"), dict),
                    "protocol",
                    "Expected correlated mock submission and wire representation",
                )
                if self._active is not None:
                    reject(
                        response.get("preempted_sequence")
                        == self._active.request.sequence,
                        "protocol",
                        "Mock endpoint did not report the previous request disposition",
                    )
                    self._transition(
                        self._active, ActionState.CANCELED, "preempted_by_new_request"
                    )
                record.submitted_ns = self.clock.now_ns()
                self._pending, self._active = None, record
                self._transition(
                    record,
                    ActionState.SUBMITTED,
                    detail={
                        "wire": response["wire"],
                        "admission_deadline_ns": valid_until,
                    },
                )
                state = ActionState(response["state"])
                if state is ActionState.ACKNOWLEDGED:
                    record.acknowledged_ns = self.clock.now_ns()
                self._transition(record, state)
                return self._snapshot()
            except ActionRejected as exc:
                state = (
                    ActionState.EXPIRED
                    if exc.code
                    in (
                        "expiry",
                        "action_age",
                        "authorization_expired",
                        "mock_admission_expired",
                    )
                    else ActionState.FAILED
                )
                self._disarm(exc.code, state)
                raise

    def send_action(self, action: Action) -> Observation:
        """Submit only this part's typed request, returning a mock-only disposition."""
        with self._lock:
            if not isinstance(action, Mapping) or set(action) != {self.ACTION_FIELD}:
                self._event(action, ActionState.REJECTED, "action_field")
                self._disarm("action_field")
                raise ActionRejected(
                    "action_field",
                    f"Expected only {self.ACTION_FIELD!r}; pose/joint/width actions are unsupported",
                )
            self.queue_action(action[self.ACTION_FIELD])
            snapshot = self.submit_pending()
            return {
                self.ACTION_FIELD: action[self.ACTION_FIELD],
                "mock_disposition": snapshot["mock_action"],
            }

    def poll(self) -> Observation:
        """Advance mock results/timeouts and report local state, never execution."""
        with self._lock:
            if not self._transport.is_alive:
                self._disarm(
                    "worker_terminated", ActionState.FAILED, contact_worker=False
                )
                self._transport.close()
                return self._snapshot()
            if not self.armed:
                return self._snapshot()
            now = self.clock.now_ns()
            if now >= self._authorization.deadline_ns:
                self._disarm("authorization_expired", ActionState.EXPIRED)
                return self._snapshot()
            try:
                self._readiness()
                if (
                    self._pending is not None
                    and now >= self._pending.request.deadline_ns
                ):
                    self._transition(
                        self._pending, ActionState.EXPIRED, "pending_expired"
                    )
                    self._pending = None
                if self._active is not None:
                    record = self._active
                    if now >= record.request.deadline_ns:
                        self._disarm("action_expired", ActionState.EXPIRED)
                    elif (
                        record.state is ActionState.ACCEPTANCE_UNKNOWN
                        and self.KIND == ROBOTIQ_KIND
                        and now - record.submitted_ns
                        >= self._profile.acknowledgement_timeout_ns
                    ):
                        self._disarm("mock_ack_timeout", ActionState.FAILED)
                    elif (
                        record.state is ActionState.ACKNOWLEDGED
                        and now - record.acknowledged_ns
                        >= self._profile.result_timeout_ns
                    ):
                        self._disarm("mock_result_timeout", ActionState.FAILED)
                    else:
                        response = self._rpc("poll")
                        reject(
                            response.get("sequence") == record.request.sequence,
                            "protocol",
                            "Mock result sequence mismatch",
                        )
                        reject(
                            response.get("state")
                            in {
                                "mock_completed",
                                "mock_canceled",
                                "failed",
                                "expired",
                                "mock_acknowledged",
                                "acceptance_unknown",
                            },
                            "protocol",
                            "Unknown mock result disposition",
                        )
                        state = ActionState(response["state"])
                        if state is ActionState.COMPLETED:
                            reject(
                                isinstance(response.get("result"), dict),
                                "protocol",
                                "Expected structured mock result",
                            )
                            self._transition(
                                record, state, detail={"result": response["result"]}
                            )
                            self._active = None
                        elif state in (
                            ActionState.CANCELED,
                            ActionState.FAILED,
                            ActionState.EXPIRED,
                        ):
                            self._transition(record, state, "mock_terminal")
                            self._active = None
                            self._disarm("mock_terminal")
                        else:
                            reject(
                                state is record.state,
                                "protocol",
                                "Unexpected mock lifecycle transition",
                            )
            except ActionRejected as exc:
                self._disarm(exc.code, ActionState.FAILED)
            return self._snapshot()

    def _snapshot(self) -> Observation:
        measured = (
            None
            if self._last is None
            else json.loads(self._last.request.observation.snapshot_json)
        )
        return {
            "mock_action": {
                "schema_version": 1,
                "offline": True,
                "physical_execution": False,
                "ros_domain_id": None,
                "worker_alive": self._transport.is_alive,
                "armed": self.armed,
                "session_id": self.session_id,
                "authority_id": self.authority_id,
                "clock_domain": self.clock.domain,
                "pending_sequence": None
                if self._pending is None
                else self._pending.request.sequence,
                "active_sequence": None
                if self._active is None
                else self._active.request.sequence,
                "last_disposition": None
                if self._last is None
                else self._last.state.value,
                "last_admitted_sequence": self._last_sequence,
                "events_evicted": self._event_count - len(self._events),
            },
            "measured": measured,
        }

    def get_state(self) -> Observation:
        """Return the local mock lifecycle and associated 2A observation provenance."""
        return self.poll()

    def get_observation(self) -> Observation:
        """Read through native per-owner snapshots."""
        return self.read_state()


class FR3OfflineAction(OfflineROS2ActionPart):
    """Six normalized measured-FK offsets, admitted only to a local mock."""

    KIND = FR3_KIND
    ACTION_FIELD = "normalized_pose_offset"


class RobotiqOfflineAction(OfflineROS2ActionPart):
    """One normalized closure request, with no force or metre-width promise."""

    KIND = ROBOTIQ_KIND
    ACTION_FIELD = "normalized_closure"
