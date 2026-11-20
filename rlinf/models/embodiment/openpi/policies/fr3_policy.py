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

"""FR3 structural input transform with no checkpoint or executable action path."""

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.data.datasets.openpi.fr3_contract import (
    CONTRACT_ID,
    CONTRACT_PATH,
    load_fr3_contract,
    verify_openpi_installation,
)
from rlinf.data.storage.fr3_validation import _robot, require


@dataclass(frozen=True)
class FR3Inputs:
    """Copy verified native representations and prepare synthetic image probes.

    This callable follows RLinf's DataTransformFn protocol. Its state is structural
    data; model targets and authentic language remain explicitly unavailable.
    """

    contract_path: str | Path = CONTRACT_PATH
    state_dimension: int = 42
    action_dimension: int = 7
    _contract: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        require(
            self.state_dimension == 42 and self.action_dimension == 7,
            "FR3 structural dimensions must be 42 state / 7 request; legacy 19D and model padding are unsupported",
        )
        contract = load_fr3_contract(self.contract_path)
        verify_openpi_installation(contract)
        object.__setattr__(self, "_contract", contract)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        contract, context = self._contract, data["fr3_context"]
        require(
            context["physical_execution"] is False and context["synthetic"] is True,
            "FR3 inputs require synthetic data and physical_execution=false",
        )
        require(
            context["state_fields"] == [f["name"] for f in contract["state"]["fields"]],
            "Incorrect FR3 42D feature ordering or missing features",
        )
        require(
            context["joint_names"] == contract["state"]["joint_names"],
            "Incorrect FR3 joint ordering",
        )
        require(
            context["action_kinds"] == contract["actions"]["kinds"],
            "Incorrect action discriminator; expected normalized pose offset and closure",
        )
        require(
            context["action_semantics"] == contract["actions"]["semantics"],
            "Absolute/relative action or gripper semantics mismatch",
        )
        require(
            context["camera_roles"] == contract["cameras"]["physical_roles"],
            "Unqualified physical camera-role assignment",
        )
        require(
            context["frame_contract"]
            == {f["name"]: f["frame"] for f in contract["state"]["fields"]},
            "Unverified FR3 frame conversion",
        )
        state, actions = np.asarray(data["state"]), np.asarray(data["actions"])
        require(
            state.shape == (42,)
            and state.dtype == np.float32
            and np.isfinite(state).all(),
            "Expected finite float32 FR3 state [42]",
        )
        require(
            abs(float(np.linalg.norm(state[35:39])) - 1) <= 1e-5,
            "Expected unit xyzw FR3 quaternion",
        )
        require(0 <= state[28] <= 1, "Invalid measured normalized closure")
        observation = context["observation"]
        for part in ("arm", "gripper"):
            _robot(observation, part)
        measured = {**observation["arm"]["state"], **observation["gripper"]["state"]}
        expected_state = np.concatenate(
            [np.atleast_1d(measured[f["name"]]) for f in contract["state"]["fields"]]
        ).astype(np.float32)
        require(
            np.array_equal(state, expected_state),
            "FR3 state values/order differ from named native observations",
        )
        require(
            actions.ndim == 2
            and actions.shape[1] == 7
            and actions.dtype == np.float32
            and np.isfinite(actions).all(),
            "Expected finite float32 controller requests [H,7]",
        )
        padding = np.asarray(data["actions_is_pad"])
        require(
            padding.shape == (actions.shape[0],) and padding.dtype == np.bool_,
            "Expected Boolean action padding mask [H]",
        )
        require(
            (np.abs(actions[:, :6]) <= 1).all()
            and ((actions[:, 6] >= 0) & (actions[:, 6] <= 1)).all(),
            "Normalized controller request outside representation range",
        )
        typed = context["typed_requests"]
        require(
            [typed[k]["kind"] for k in ("arm", "end_effector")]
            == contract["actions"]["kinds"],
            "Typed action discriminator differs from recorded request semantics",
        )
        require(
            np.array_equal(
                actions[0],
                np.array(
                    typed["arm"]["values"] + typed["end_effector"]["values"],
                    dtype=np.float32,
                ),
            ),
            "First controller request differs from typed provenance",
        )
        raw_images = {}
        for name, key in contract["cameras"]["native_slots"].items():
            image = np.asarray(data[key])
            require(
                image.ndim == 3
                and image.shape[-1] == 3
                and min(image.shape[:2]) > 0
                and image.dtype == np.uint8,
                f"Expected RGB uint8 HWC camera {name}",
            )
            raw_images[name] = image.copy()
            require(
                hashlib.sha256(image.tobytes()).hexdigest()
                == observation["cameras"][name]["sha256"],
                f"Camera image association differs from native {name} record",
            )
        masks = {
            slot: sensor is not None
            for slot, sensor in contract["cameras"]["structural_probe_slots"].items()
        }
        images = {
            slot: raw_images[sensor].copy()
            if sensor
            else np.zeros_like(raw_images["d435i"])
            for slot, sensor in contract["cameras"]["structural_probe_slots"].items()
        }
        import jax
        from openpi.models.model import Observation
        from openpi.transforms import ResizeImages

        require(
            jax.default_backend() == "cpu",
            "FR3 structural preprocessing requires JAX_PLATFORMS=cpu",
        )
        resized = ResizeImages(224, 224)({"image": images})["image"]
        probe = Observation.from_dict(
            {
                "image": {k: np.asarray(v) for k, v in resized.items()},
                "image_mask": {k: np.asarray(v, dtype=bool) for k, v in masks.items()},
                "state": state.copy(),
            }
        )
        return {
            "contract_id": CONTRACT_ID,
            "native_state": state.copy(),
            "state": state.copy(),
            "recorded_controller_requests": actions.copy(),
            "model_training_target": None,
            "predicted_model_action": None,
            "prompt": None,
            "raw_images": raw_images,
            "image": {k: np.asarray(v).copy() for k, v in probe.images.items()},
            "image_mask": {
                k: np.asarray(v).copy() for k, v in probe.image_masks.items()
            },
            "actions_is_pad": padding.copy(),
            "provenance": copy.deepcopy(context),
            "checkpoint_status": "CHECKPOINT_UNVERIFIED",
            "physical_execution": False,
        }

    def require_model_ready(self) -> None:
        """Reject model execution until a new contract qualifies the checkpoint."""
        raise ValueError(
            "CHECKPOINT_UNVERIFIED: no qualified model state/action target, camera roles, instruction, normalization or checkpoint assets"
        )


@dataclass(frozen=True)
class FR3Outputs:
    """Reject predictions; recorded controller requests are not policy outputs."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        raise ValueError(
            "CHECKPOINT_UNVERIFIED: predicted actions cannot become FR3 controller requests; no physical transport exists"
        )
