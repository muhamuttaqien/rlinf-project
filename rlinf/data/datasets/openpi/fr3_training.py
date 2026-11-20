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

"""Pinned checkpoint assets and synthetic FR3 training-schema probes."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.data.datasets.openpi.fr3 import FR3OfflineDataset
from rlinf.data.datasets.openpi.fr3_contract import (
    load_fr3_contract,
    verify_openpi_installation,
)
from rlinf.data.storage.fr3_synthetic import read_json
from rlinf.data.storage.fr3_validation import artifact, require

CANDIDATE_PATH = Path(__file__).with_name("fr3_pi05_candidate_v1.json")
CANDIDATE_SHA256 = "b194c9682975f96fe2573928056a1f47ce87986ab8155cc96073ab1e25b8b634"


def load_checkpoint_candidate(path: str | Path = CANDIDATE_PATH) -> dict:
    """Load the immutable object-generation set and reviewed inspection contract."""
    candidate = read_json(Path(path))
    digest = hashlib.sha256(
        json.dumps(
            candidate, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    require(digest == CANDIDATE_SHA256, "Checkpoint candidate fingerprint mismatch")
    return candidate


def verify_checkpoint_sources(candidate: dict) -> dict:
    """Check the installed OpenPI pin and the native RLinf inspection sources."""
    runtime = verify_openpi_installation(load_fr3_contract())
    root = Path(__file__).resolve().parents[4]
    for relative, expected in candidate["local_source_sha256"].items():
        require(
            hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected,
            f"Unqualified RLinf checkpoint integration source: {relative}",
        )
    return runtime


def _asset(root: Path, entry: dict) -> bytes:
    path = artifact(root, entry["bucket"] + "/" + entry["name"])
    require(path.stat().st_size == entry["size"], f"Asset size mismatch: {path.name}")
    data = path.read_bytes()
    require(
        hashlib.sha256(data).hexdigest() == entry["sha256"],
        f"Asset hash mismatch: {entry['name']}",
    )
    return data


def inspect_checkpoint_assets(root: str | Path) -> dict:
    """Inspect only pinned small metadata/tokenizer files; never restore parameters."""
    candidate, root = load_checkpoint_candidate(), Path(root).resolve()
    verified, dimensions = {}, {}
    shapes = {}
    for entry in candidate["objects"]:
        if not entry.get("downloaded_for_inspection"):
            continue
        data = _asset(root, entry)
        verified[entry["name"]] = entry["sha256"]
        parsed = json.loads(data)
        if entry["name"].endswith("norm_stats.json"):
            stats = parsed["norm_stats"]
            require(
                set(stats) == {"state", "actions"}, "Unexpected checkpoint statistics"
            )
            for name, values in stats.items():
                require(
                    set(values) == {"mean", "std", "q01", "q99"},
                    f"Incomplete checkpoint statistics: {name}",
                )
                arrays = {k: np.asarray(v) for k, v in values.items()}
                require(
                    all(
                        v.shape == (8,) and np.isfinite(v).all()
                        for v in arrays.values()
                    )
                    and (arrays["std"] > 0).all()
                    and (arrays["q99"] > arrays["q01"]).all(),
                    f"Invalid checkpoint statistics: {name}",
                )
            dimensions[entry["name"].split("/")[-2]] = {"state": 8, "actions": 8}
        if entry["name"].endswith("array_metadatas/process_0"):
            shapes = {
                e["array_metadata"]["param_name"]: e["array_metadata"]["write_shape"]
                for e in parsed["array_metadatas"]
            }
    require(
        len(shapes) == 51
        and shapes["params.action_in_proj.kernel"] == [32, 1024]
        and shapes["params.action_out_proj.kernel"] == [1024, 32]
        and shapes["params.PaliGemma.llm.embedder.input_embedding"] == [257152, 2048]
        and not any("state_proj" in name for name in shapes),
        "Checkpoint metadata disagrees with PI05 architecture",
    )
    tokenizer = _asset(root, candidate["tokenizer"])
    return {
        "status": "PASS_SMALL_ASSET_IDENTITIES_ONLY",
        "files_verified": len(verified) + 1,
        "downloaded_asset_bytes": sum(
            e["size"]
            for e in candidate["objects"]
            if e.get("downloaded_for_inspection")
        )
        + len(tokenizer),
        "normalization_dimensions": dimensions,
        "fr3_normalization_compatibility": "INCOMPATIBLE_42_STATE_7_ACTION_VS_8_8",
        "tensor_metadata_arrays": len(shapes),
        "action_in_projection": shapes["params.action_in_proj.kernel"],
        "action_out_projection": shapes["params.action_out_proj.kernel"],
        "state_projection_present": False,
        "weights_restored": False,
    }


def tokenize_fr3_fixture(
    root: str | Path, state: np.ndarray, prompt: str, *, scope: str
) -> dict[str, Any]:
    """Use the native tokenizer with pinned local bytes and reject silent truncation.

    State and text are explicit schema fixtures, never replacement training labels.
    OPENPI_DATA_HOME must point at the already populated asset directory.
    """
    candidate, root = load_checkpoint_candidate(), Path(root).resolve()
    verify_checkpoint_sources(candidate)
    require(
        importlib.metadata.version("sentencepiece")
        == candidate["tokenizer"]["sentencepiece_version"],
        "Unqualified SentencePiece version",
    )
    require(
        scope == "synthetic_math_only",
        "Tokenizer probe requires synthetic fixture scope",
    )
    require(
        isinstance(prompt, str) and bool(prompt.strip()),
        "Missing explicit synthetic tokenizer instruction",
    )
    state = np.asarray(state)
    require(
        state.shape == (42,)
        and state.dtype == np.float32
        and np.isfinite(state).all()
        and (np.abs(state) <= 1).all(),
        "Tokenizer fixture requires finite normalized float32 state [42] in [-1,1]",
    )
    require(
        os.environ.get("OPENPI_DATA_HOME") is not None
        and Path(os.environ["OPENPI_DATA_HOME"]).resolve() == root,
        "Set OPENPI_DATA_HOME to the verified local asset directory; downloads are disabled",
    )
    data = _asset(root, candidate["tokenizer"])
    import sentencepiece
    from openpi.models.tokenizer import PaligemmaTokenizer

    processor = sentencepiece.SentencePieceProcessor(model_proto=data)
    require(processor.vocab_size() == 257152, "Unexpected PaliGemma vocabulary")
    bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    text = f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: "
    raw = processor.encode(text, add_bos=True)
    maximum = candidate["architecture"]["max_token_len"]
    require(
        len(raw) <= maximum, f"Tokenizer truncation prohibited: {len(raw)} > {maximum}"
    )
    tokens, mask = PaligemmaTokenizer(max_len=maximum).tokenize(prompt, state)
    require(
        np.array_equal(tokens[mask], raw)
        and int(mask.sum()) == len(raw)
        and tokens.shape == mask.shape == (maximum,),
        "Native tokenizer disagrees with pinned PI05 prompt representation",
    )
    return {
        "tokens": tokens.astype(np.int32),
        "mask": mask.astype(np.bool_),
        "token_count": len(raw),
        "vocabulary_size": processor.vocab_size(),
        "scope": scope,
        "truncated": False,
        "checkpoint_semantics_qualified": False,
    }


def build_fr3_request_target_proposal(
    dataset: FR3OfflineDataset, index: int, *, require_full_window: bool = False
) -> dict[str, Any]:
    """Prepare lossless request-imitation schema data, separate from model targets.

    The consumer must implement the returned per-element mask before any future
    training. Native SFT does not currently consume that mask.
    """
    require(
        isinstance(dataset, FR3OfflineDataset), "Expected audited FR3OfflineDataset"
    )
    sample = dataset[index]
    require(
        sample["model_training_target"] is None
        and sample["physical_execution"] is False,
        "Model target or physical execution is unqualified",
    )
    if require_full_window:
        require(
            not sample["actions_is_pad"].any(),
            "Missing future actions in full-window proposal",
        )
    from openpi.transforms import PadStatesAndActions

    padded = PadStatesAndActions(32)(
        {
            "state": sample["native_state"].copy(),
            "actions": sample["recorded_controller_requests"].copy(),
        }
    )
    require(
        np.array_equal(padded["state"], sample["native_state"]),
        "Native state must remain 42D",
    )
    mask = np.zeros_like(padded["actions"], dtype=bool)
    mask[:, :7] = ~sample["actions_is_pad"][:, None]
    return {
        "schema": "fr3_request_imitation_proposal_v1",
        "scope": "synthetic_schema_only_not_checkpoint_target",
        "native_state": padded["state"],
        "recorded_requests": sample["recorded_controller_requests"].copy(),
        "proposed_actions": padded["actions"],
        "proposed_loss_mask": mask,
        "provenance": sample["provenance"],
        "model_training_target": None,
        "normalization_applied": False,
        "native_sft_mask_supported": False,
        "physical_execution": False,
    }


def assess_fr3_training_requirements(dataset: FR3OfflineDataset) -> dict:
    """Report existing archive/RECAP bindings and missing future collection assets."""
    candidate = load_checkpoint_candidate()
    missing = dict.fromkeys(candidate["training_metadata"]["required"], 0)
    episodes = dataset.metadata_document()["episodes"]
    for record in episodes:
        missing["original_instruction"] += int(
            record["task"] is None or record["task"]["origin"] != "recorded_original"
        )
        missing["task_id_version"] += int(record["task"] is None)
        missing["episode_split"] += int(record["split"] == "unassigned")
        for key in (
            "collection_session",
            "camera_configuration",
            "normalization_provenance",
        ):
            missing[key] += 1
    return {
        "schema": "fr3_training_requirements_v1",
        "episodes": len(episodes),
        "source_manifest_sha256": dataset.source_manifest_sha256,
        "candidate_id": candidate["candidate_id"],
        "candidate_association": "offline_candidate_only",
        "available_native_fields": [
            "sessions",
            "config_sha256",
            "action_schema_version",
            "observation_schema_version",
        ],
        "resolved_configuration_sha256": dataset.manifest["config_sha256"],
        "observation_schema_version": dataset.manifest["observation_schema_version"],
        "action_schema_version": dataset.manifest["action_schema_version"],
        "action_authority_sessions": {
            e["episode_id"]: e["sessions"] for e in dataset._episodes
        },
        "missing_episode_counts": missing,
        "camera_identity_is_configuration_qualification": False,
        "recap": dataset.metadata_report,
        "training_readiness": "UNQUALIFIED_SYNTHETIC_ONLY",
    }


def require_fr3_training_ready() -> None:
    """Keep structural probes outside all training/inference execution paths."""
    raise ValueError(
        "CHECKPOINT_UNVERIFIED: FR3 statistics, original instructions, camera semantics, "
        "weight conversion and padding-loss-mask integration remain unqualified"
    )
