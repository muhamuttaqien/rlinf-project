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

"""Immutable training-source evolution and bounded checkpoint/runtime audits."""

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import numpy as np
import torch
from packaging.requirements import Requirement

from rlinf.data.storage.fr3_validation import require

TRAINING_CONTRACT_PATH = Path(__file__).with_name("fr3_training_v1.json")
TRAINING_CONTRACT_SHA256 = (
    "f96072979c539aee2afef0a00403056b9ecf283e7bcc78d4dca4e37572776a31"
)


def load_training_contract(path: str | Path = TRAINING_CONTRACT_PATH) -> dict:
    """Load the reviewed source/schema profile without changing older contracts."""
    profile = json.loads(Path(path).read_text())
    require(
        hashlib.sha256(
            json.dumps(
                profile, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        == TRAINING_CONTRACT_SHA256,
        "FR3 training contract fingerprint mismatch",
    )
    return profile


def verify_training_sources() -> dict:
    """Qualify the exact native loss/worker source successor of Milestone 7."""
    profile = load_training_contract()
    root = Path(__file__).resolve().parents[4]
    for name, digest in profile["native_source_sha256"].items():
        require(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest,
            f"Unqualified masked SFT source: {name}",
        )
    return profile


def runtime_compatibility_report() -> dict:
    """Audit package requirements/file ownership; never install or patch packages."""
    distributions = {
        n: metadata.version(n)
        for n in (
            "rlinf-openpi",
            "rlinf-transformer-openpi",
            "transformers",
            "torch",
            "jax",
            "jaxlib",
            "orbax-checkpoint",
            "tokenizers",
            "numpy",
            "sentencepiece",
        )
    }
    violations = []
    for owner in ("rlinf-openpi", "rlinf-transformer-openpi", "transformers"):
        for raw in metadata.requires(owner) or []:
            r = Requirement(raw)
            if r.marker is not None and not r.marker.evaluate({"extra": ""}):
                continue
            try:
                installed = metadata.version(r.name)
            except metadata.PackageNotFoundError:
                installed = None
            if installed is None or not r.specifier.contains(installed):
                violations.append(
                    {"owner": owner, "requirement": str(r), "installed": installed}
                )
    shared = []
    for name in (
        "transformers/__init__.py",
        "transformers/models/gemma/modeling_gemma.py",
        "transformers/models/paligemma/modeling_paligemma.py",
    ):
        owners = {}
        for owner in ("rlinf-transformer-openpi", "transformers"):
            d = metadata.distribution(owner)
            f = next((f for f in d.files or [] if str(f) == name), None)
            if f is not None:
                owners[owner] = {
                    "record_hash": str(f.hash),
                    "actual_sha256": hashlib.sha256(
                        d.locate_file(f).read_bytes()
                    ).hexdigest(),
                }
        shared.append({"path": name, "owners": owners})
    return {
        "status": "UNQUALIFIED",
        "isolated_environment_required": True,
        "distributions": distributions,
        "requirement_violations": violations,
        "overlapping_file_ownership": shared,
        "constraints": load_training_contract()["runtime_constraints"],
        "large_dependency_installation_performed": False,
        "full_model_import_or_execution_qualification": False,
    }


def convert_projection_fixture(
    kernel: np.ndarray, bias: np.ndarray, *, scope: str
) -> dict:
    """Exercise the official projection transpose/identity rule on synthetic arrays."""
    require(
        scope == "synthetic_math_only", "Only synthetic projection fixtures are allowed"
    )
    kernel, bias = np.asarray(kernel), np.asarray(bias)
    require(
        kernel.ndim == 2
        and bias.shape == (kernel.shape[1],)
        and kernel.dtype == bias.dtype == np.float32
        and np.isfinite(kernel).all()
        and np.isfinite(bias).all(),
        "Invalid synthetic projection shape/dtype/values",
    )
    return {
        "action_in_proj.weight": torch.from_numpy(kernel.copy()).T.contiguous(),
        "action_in_proj.bias": torch.from_numpy(bias.copy()),
    }


def audit_conversion_fixture(output_root: str | Path) -> dict:
    """Round-trip a tiny labeled tensor fixture through native safetensors loading."""
    from safetensors.torch import load_file, save_file

    from rlinf.models.embodiment.openpi.checkpoint import load_base_safetensors

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    kernel = np.arange(12, dtype=np.float32).reshape(3, 4)
    bias = np.arange(4, dtype=np.float32)
    converted = convert_projection_fixture(kernel, bias, scope="synthetic_math_only")
    path = output_root / "synthetic_projection_fixture.safetensors"
    save_file(
        converted,
        path,
        metadata={"scope": "synthetic_math_only", "eligible_checkpoint": "false"},
    )
    independently_loaded = load_file(path, device="cpu")
    require(
        set(independently_loaded) == set(converted),
        "Converted fixture keys changed on reload",
    )
    toy = torch.nn.Module()
    toy.action_in_proj = torch.nn.Linear(3, 4)
    load_base_safetensors(toy, path)
    for name, value in toy.state_dict().items():
        require(
            torch.equal(value, converted[name])
            and torch.equal(value, independently_loaded[name]),
            "Converted fixture tensor values changed",
        )
    require(
        np.array_equal(toy.action_in_proj.weight.detach().numpy().T, kernel),
        "Synthetic projection inverse differs",
    )
    return {
        "status": "PASS_SYNTHETIC_PROJECTION_SUBSET",
        "full_weight_conversion": "WEIGHT_CONVERSION_NOT_EXECUTED",
        "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fixture_path": str(path),
        "source_kernel_shape": [3, 4],
        "destination_weight_shape": [4, 3],
        "independent_native_reload": True,
        "eligible_checkpoint": False,
        "recipe": load_training_contract()["conversion"],
    }
