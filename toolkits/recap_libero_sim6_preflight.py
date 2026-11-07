# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Read-only preflight for LIBERO-10 Task 0 RECAP CFG policy training.

Usage:
    python -m toolkits.recap_libero_sim6_preflight

Checks the two dataset directories, final SIM-5 advantage labels, PI0.5
weights and the normalization actually selected by OpenPI. Does not train.
"""

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/offline_rl/config/experiments/libero10_task0/cfg_rl_openpi.yaml"
)


def preflight(config_path: Path = CONFIG_PATH) -> None:
    cfg = OmegaConf.load(config_path)
    sources = list(cfg.data.train_data_paths)
    expected = {"sft", "rollout"}
    actual = {str(x.type).lower() for x in sources}
    if actual != expected or len(sources) != 2:
        raise ValueError(f"Expected SFT + rollout sources, got {actual}")

    model_path = Path(cfg.actor.model.model_path).expanduser()
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"Base pi05 checkpoint missing model.safetensors in {model_path}"
        )
    print(f"PASS: model checkpoint: {model_path}")

    sft = next(x for x in sources if str(x.type).lower() == "sft")
    rollout = next(x for x in sources if str(x.type).lower() == "rollout")
    for ds in sources:
        root = Path(ds.dataset_path).expanduser()
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Dataset metadata missing: {root}")
        print(f"PASS: {ds.type} dataset: {root}")

    path = (
        Path(rollout.dataset_path)
        / "meta"
        / f"advantages_{cfg.data.advantage_tag}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"SIM-5 advantages missing: {path}")

    labels = pd.read_parquet(path, columns=["advantage"])
    if labels["advantage"].isna().any():
        raise ValueError(f"Advantage labels contain null values: {path}")
    rate = float(labels["advantage"].mean())
    if not 0 < rate < 1:
        raise ValueError(f"Rollout labels must include positives and negatives: {rate}")
    print(f"PASS: rollout labels: {len(labels):,} rows, {rate:.2%} positive")
    print("PASS: SFT demonstrations treated as positive (no sidecar required)")

    # Verify the actual OpenPI normalization (the base pi05 norm_stats.json
    # can differ from LIBERO's 8-dim state / 7-dim action assets).
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    train = get_openpi_config(
        cfg.actor.model.openpi.config_name,
        model_path=str(model_path),
        repo_id=sft.dataset_path,
        data_kwargs=cfg.actor.model.get("openpi_data", None),
    )
    resolved = train.data.create(train.assets_dirs, train.model)
    norms = resolved.norm_stats or {}
    if not all(k in norms for k in ("state", "actions")):
        raise ValueError(
            "LIBERO norm_stats missing state/actions. Provide the local "
            "LIBERO norm_stats.json with "
            "actor.model.openpi_data.norm_stats_path=/absolute/path/norm_stats.json"
        )
    state_dim = len(norms["state"].mean)
    action_dim = len(norms["actions"].mean)
    if (state_dim, action_dim) != (8, 7):
        raise ValueError(
            f"Wrong normalization for LIBERO: state={state_dim}, "
            f"actions={action_dim}; expected 8 and 7. Set "
            "actor.model.openpi_data.norm_stats_path to the LIBERO norm_stats.json, "
            "not pi05_base norm_stats.json."
        )
    print("PASS: LIBERO normalization: state=8, actions=7")
    print("SIM-6 PRE-FLIGHT PASSED")


def main() -> int:
    preflight()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
