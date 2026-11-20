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

"""Immutable source and representation pins for offline FR3 OpenPI inspection."""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import numpy as np

from rlinf.data.storage.fr3_synthetic import read_json
from rlinf.data.storage.fr3_validation import require

CONTRACT_PATH = Path(__file__).with_name("fr3_contract_v1.json")
CONTRACT_SHA256 = "e731df69aee38bed82e832f8108f43787a53670e6d621031fbc5ec3565de614c"
CONTRACT_ID = "fr3_openpi_structural_v1"


def load_fr3_contract(path: Path | str = CONTRACT_PATH) -> dict:
    """Load the reviewed structural contract; reject semantic changes under its ID."""
    contract = read_json(Path(path))
    digest = hashlib.sha256(
        json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    require(
        digest == CONTRACT_SHA256,
        "FR3 contract fingerprint mismatch; expected the reviewed v1 contract",
    )
    return contract


def verify_openpi_installation(contract: dict) -> dict:
    """Verify the installed package and complete Python source inventory before use."""
    require(
        importlib.metadata.version("rlinf-openpi") == contract["openpi"]["version"],
        "Unqualified OpenPI distribution version",
    )
    spec = importlib.util.find_spec("openpi")
    require(
        spec is not None and spec.origin is not None,
        "Pinned OpenPI package unavailable",
    )
    root = Path(spec.origin).parent
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }
    require(
        files == contract["openpi"]["python_files_sha256"],
        "Installed OpenPI source differs from the qualified source pin",
    )
    for package in ("lerobot", "numpy", "jax", "jaxlib"):
        require(
            importlib.metadata.version(package) == contract["dependencies"][package],
            f"Unqualified {package} version",
        )
    return {
        "distribution": "rlinf-openpi",
        "version": contract["openpi"]["version"],
        "source_tree_sha256": contract["openpi"]["source_tree_sha256"],
        "python_files_verified": len(files),
        "installed_git_revision": None,
    }


def normalization_fixture_check(
    state: np.ndarray, actions: np.ndarray, fixture: dict
) -> dict:
    """Exercise native normalization mathematics with explicitly synthetic statistics.

    These values never become checkpoint assets or model training targets.
    """
    contract = load_fr3_contract()
    verify_openpi_installation(contract)
    require(
        fixture.get("schema") == "fr3_normalization_fixture_v1"
        and fixture.get("scope") == "synthetic_math_only"
        and fixture.get("contract_id") == CONTRACT_ID,
        "Expected synthetic-only normalization fixture provenance",
    )
    require(
        fixture.get("method") in ("zscore", "quantile"),
        "Unsupported normalization method",
    )
    data = {
        "state": np.array(state, copy=True),
        "actions": np.array(actions, copy=True),
    }
    require(
        data["state"].shape == (42,)
        and data["actions"].ndim == 2
        and data["actions"].shape[1] == 7,
        "Normalization fixture requires state [42] and actions [H,7]",
    )
    require(
        all(v.dtype == np.float32 and np.isfinite(v).all() for v in data.values()),
        "Expected finite float32 normalization inputs",
    )
    statistics = fixture.get("statistics")
    require(
        isinstance(statistics, dict) and set(statistics) == {"state", "actions"},
        "Missing normalization assets for state/actions",
    )
    from openpi.shared.normalize import NormStats
    from openpi.transforms import Normalize, Unnormalize

    stats = {}
    for name, dim in (("state", 42), ("actions", 7)):
        entry = statistics[name]
        expected = {"mean", "std"} | (
            {"q01", "q99"} if fixture["method"] == "quantile" else set()
        )
        require(
            isinstance(entry, dict) and set(entry) == expected,
            f"Missing/unsupported normalization statistics: {name}",
        )
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in entry.items()}
        require(
            all(a.shape == (dim,) and np.isfinite(a).all() for a in arrays.values()),
            f"Invalid normalization dimensions/values: {name}",
        )
        require(
            (arrays["std"] > 0).all(),
            f"Expected positive normalization standard deviations: {name}",
        )
        if fixture["method"] == "quantile":
            require(
                (arrays["q99"] > arrays["q01"]).all(),
                f"Expected increasing normalization quantiles: {name}",
            )
        stats[name] = NormStats(**arrays)
    normalized = Normalize(
        stats, use_quantiles=fixture["method"] == "quantile", strict=True
    )(data)
    restored = Unnormalize(stats, use_quantiles=fixture["method"] == "quantile")(
        normalized
    )
    require(
        np.allclose(restored["state"], state, atol=1e-5)
        and np.allclose(restored["actions"], actions, atol=1e-5),
        "Native normalization fixture round-trip mismatch",
    )
    return {
        "status": "PASS_SYNTHETIC_MATH_ONLY",
        "method": fixture["method"],
        "scope": "synthetic_math_only",
        "checkpoint_qualification": "UNQUALIFIED",
        "normalized": normalized,
        "restored": restored,
    }
