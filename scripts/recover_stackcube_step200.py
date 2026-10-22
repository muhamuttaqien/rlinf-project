# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recover model-only Pi0.5 weights from an unfinished PyTorch DCP checkpoint.

RLinf saves the distributed training state before exporting full_weights.pt.
When storage fills up, the model shard may exist without DCP .metadata or
full_weights.pt. Recovery uses original step-200 metadata, if present.
With explicit --use-step100-layout, it can *provisionally* reuse step-100
byte offsets from the same training run, validating every model tensor,
then comparing the complete model key/shape/dtype contract to step 100.

Inference of missing metadata is not independently provable: a matching
shape and dtype cannot rule out swapped same-shaped tensors. Preserve both
originals and evaluate any recovered candidate as provisional.

This tool requires PyTorch, but does not initialize GPUs, Ray or ManiSkill.
Any recovered weights are written to a separate directory on /data1.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

MODEL_PREFIX = "fsdp_checkpoint.model."
RESULT_ROOT = Path(
    "/data1/Muttaqien/rlinf_data/results/"
    "maniskill_stackcube_panda_pi05_state_sft"
)
OUTPUT = Path(
    "/data1/Muttaqien/rlinf_data/recovery/state_step200/"
    "actor/model_state_dict/full_weights.pt"
)


def read_metadata(path: Path) -> Any:
    """Only use metadata from the user's own trusted RLinf checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing DCP .metadata: {path}")
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def model_entries(metadata: Any) -> list[tuple[str, Any, Any]]:
    """Single-GPU: require one complete chunk for every model tensor."""
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    entries = []
    for fqn, meta in metadata.state_dict_metadata.items():
        if not fqn.startswith(MODEL_PREFIX):
            continue
        if not isinstance(meta, TensorStorageMetadata):
            raise ValueError(f"Non-tensor model entry: {fqn}")
        stored = [
            (index, item)
            for index, item in metadata.storage_data.items()
            if index.fqn == fqn
        ]
        if len(stored) != 1 or len(meta.chunks) != 1:
            raise ValueError(f"Model tensor is sharded: {fqn}")
        _index, item = stored[0]
        chunk = meta.chunks[0]
        if (
            any(int(n) != 0 for n in chunk.offsets)
            or tuple(chunk.sizes) != tuple(meta.size)
        ):
            raise ValueError(f"Partial model tensor unsupported: {fqn}")
        entries.append((fqn[len(MODEL_PREFIX) :], meta, item))
    if not entries:
        raise ValueError("No fsdp_checkpoint.model.* tensors found in DCP metadata")
    return entries


def ensure_complete_ranges(
    dcp_dir: Path, entries: list[tuple[str, Any, Any]]
) -> dict[str, int]:
    """Reject any model tensor extending beyond the available .distcp shard."""
    sizes = {}
    for key, _meta, storage in entries:
        path = Path(storage.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe shard path: {path}")
        shard = dcp_dir / path
        if not shard.is_file():
            raise FileNotFoundError(f"Missing shard {shard}")
        length = sizes.setdefault(str(path), shard.stat().st_size)
        offset = int(storage.offset)
        size = int(storage.length)
        if offset < 0 or size <= 0:
            raise ValueError(f"Invalid storage range for {key}")
        if offset + size > length:
            raise ValueError(
                f"Incomplete shard for {key}: needs ending byte "
                f"{offset + size:,}, but shard has {length:,} bytes"
            )
    return sizes


def load_tensor(dcp_dir: Path, meta: Any, storage: Any):
    """Use PyTorch's bounded DCP tensor view, not a raw file-wide torch.load."""
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    reader = FileSystemReader(str(dcp_dir))
    with (dcp_dir / storage.relative_path).open("rb") as stream:
        view = reader._slice_file(stream, storage)
        tensor = torch.load(view, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Non-tensor DCP payload at {storage.offset}")
    if tuple(tensor.shape) != tuple(meta.size):
        raise ValueError(
            f"Tensor shape mismatch {tuple(tensor.shape)} != {tuple(meta.size)}"
        )
    if tensor.dtype != meta.properties.dtype:
        raise ValueError(f"Tensor dtype mismatch: {tensor.dtype}")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise ValueError("Recovered model tensor contains NaN or Inf")
    return tensor


def probe_layout(dcp_dir: Path, entries: list[tuple[str, Any, Any]]) -> None:
    for key, meta, storage in (
        entries[0],
        entries[len(entries) // 2],
        entries[-1],
    ):
        try:
            load_tensor(dcp_dir, meta, storage)
        except Exception as exc:
            raise ValueError(f"Byte layout probe failed for {key}: {exc}") from exc


def normalize_keys(state: dict[str, Any]) -> dict[str, Any]:
    """Mirror RLinf's checkpoint wrapper-key normalization for comparison."""
    pi0_prefixes = (
        "llm.", "img.", "action_in_proj.", "action_out_proj.",
        "time_mlp_in.", "time_mlp_out.", "state_proj.",
        "action_time_mlp_in.", "action_time_mlp_out.", "pointnet.",
    )
    result = {}
    for key, value in state.items():
        while True:
            for prefix in ("_fsdp_wrapped_module.", "_orig_mod.", "module."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            else:
                break
        if key.startswith("model.") and key[len("model.") :].startswith(pi0_prefixes):
            key = key[len("model.") :]
        if key in result:
            raise ValueError(f"Duplicate normalized model tensor key: {key}")
        result[key] = value
    return result


def validate_model_contract(
    recovered: dict[str, Any], step100_fullweights: Path
) -> None:
    """Check every key, dtype and shape against the known-good actor."""
    import torch

    if not step100_fullweights.is_file():
        raise FileNotFoundError(f"Missing step-100 full weights: {step100_fullweights}")
    baseline = torch.load(
        str(step100_fullweights), map_location="meta", weights_only=True
    )
    if not isinstance(baseline, dict) or not all(
        isinstance(v, torch.Tensor) for v in baseline.values()
    ):
        raise ValueError("Step-100 full weights are not a flat model state dictionary")
    expected = normalize_keys(baseline)
    observed = normalize_keys(recovered)
    if expected.keys() != observed.keys():
        missing = sorted(expected.keys() - observed.keys())[:5]
        extra = sorted(observed.keys() - expected.keys())[:5]
        raise ValueError(f"Model tensor key mismatch: missing={missing}, extra={extra}")
    for key, value in observed.items():
        old = expected[key]
        if value.shape != old.shape or value.dtype != old.dtype:
            raise ValueError(
                f"Model tensor mismatch for {key}: "
                f"{value.shape}/{value.dtype} vs {old.shape}/{old.dtype}"
            )


def run(args: argparse.Namespace) -> None:
    import torch

    root = args.results_root.expanduser().resolve()
    dcp100 = root / "checkpoints/global_step_100/actor/dcp_checkpoint"
    dcp200 = root / "checkpoints/global_step_200/actor/dcp_checkpoint"
    full100 = root / (
        "checkpoints/global_step_100/actor/model_state_dict/full_weights.pt"
    )
    out = args.output.expanduser().resolve()
    print(f"PyTorch: {torch.__version__}")
    print(f"Step 100 DCP metadata: {'FOUND' if (dcp100 / '.metadata').is_file() else 'MISSING'}")
    print(f"Step 200 DCP metadata: {'FOUND' if (dcp200 / '.metadata').is_file() else 'MISSING'}")
    print(f"Step 100 full weights: {'FOUND' if full100.is_file() else 'MISSING'}")

    has_meta200 = (dcp200 / ".metadata").is_file()
    if has_meta200:
        layout = read_metadata(dcp200 / ".metadata")
        print("Layout: original step-200 DCP metadata")
    else:
        if args.recover and not args.use_step100_layout:
            raise ValueError(
                "Step-200 metadata is absent. Explicitly pass "
                "--use-step100-layout to attempt provisional recovery."
            )
        layout = read_metadata(dcp100 / ".metadata")
        print("Layout: step-100 template (PROVISIONAL, not independently proven)")

    entries = model_entries(layout)
    print(f"Model tensor count: {len(entries)}")
    required_end = max(int(s.offset + s.length) for _, _, s in entries)
    print(f"Last required model byte: {required_end:,}")
    for shard in sorted(dcp200.glob("*.distcp")):
        print(f"Shard {shard.name}: {shard.stat().st_size:,} bytes")
    ensure_complete_ranges(dcp200, entries)
    probe_layout(dcp200, entries)
    print("Tensor ranges + three content probes: PASS")

    if not args.recover:
        print("INSPECT ONLY: no files changed")
        return
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")

    out.parent.mkdir(parents=True, exist_ok=True)
    estimated = sum(
        int(torch.tensor(md.size).prod().item())
        * torch.empty((), dtype=md.properties.dtype).element_size()
        for _, md, _ in entries
    )
    free = shutil.disk_usage(out.parent).free
    required = int(estimated * 1.25) + 2 * 1024**3
    if free < required:
        raise OSError(
            f"Recovery destination has insufficient space: "
            f"free={free:,} required={required:,}"
        )

    tensors = {}
    for n, (name, md, storage) in enumerate(entries, 1):
        tensors[name] = load_tensor(dcp200, md, storage)
        if n % 250 == 0:
            print(f"Recovered {n}/{len(entries)} tensors")
    validate_model_contract(tensors, full100)
    print("Step-100 key/shape/dtype comparison: PASS")

    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Leftover partial output exists: {partial}")
    try:
        torch.save(tensors, str(partial))
        check = torch.load(str(partial), map_location="meta", weights_only=True)
        if len(check) != len(tensors):
            raise ValueError("Saved recovery file has wrong tensor count")
        os.replace(partial, out)
    finally:
        if partial.exists():
            partial.unlink()
    print(f"RECOVERED MODEL: {out}")
    if not has_meta200:
        print("CAUTION: tensor offsets inferred from step 100; mapping is provisional.")
    print("Original step-100 and step-200 checkpoint files were not modified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--use-step100-layout", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        print(f"RECOVERY BLOCKED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
