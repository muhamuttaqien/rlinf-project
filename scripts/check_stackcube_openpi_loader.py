# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Load one RLinf/OpenPI pi0.5 SFT batch without initializing model weights."""

import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from rlinf.data.datasets.openpi import build_official_openpi_sft_dataloader


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("REPO_PATH", str(root))
    os.environ.setdefault("EMBODIED_PATH", str(root / "examples" / "sft"))
    with hydra.initialize_config_dir(
        config_dir=str(root / "examples" / "sft" / "config"),
        version_base="1.1",
    ):
        cfg = hydra.compose(config_name="stackcube_panda_sft_openpi_pi05")
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    source = resolved_cfg["data"]["train_data_paths"]
    if not (Path(source) / "meta" / "info.json").exists():
        raise FileNotFoundError(f"LeRobot dataset missing: {source}")
    stats = Path(resolved_cfg["actor"]["model"]["openpi_data"]["norm_stats_path"])
    if not stats.is_file():
        raise FileNotFoundError(
            f"OpenPI normalization missing: {stats}. "
            "Run scripts/compute_stackcube_openpi_norm_stats.py first."
        )

    loader, data_config = build_official_openpi_sft_dataloader(
        cfg, world_size=1, rank=0, data_paths=cfg.data.train_data_paths
    )
    batch = next(iter(loader))
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError(f"Expected (Observation, Actions), got {type(batch)}")
    obs, actions = batch
    print("Action batch:", tuple(actions.shape))
    if tuple(actions.shape[-2:]) != (10, 32):
        raise ValueError(f"Expected (horizon=10, model_action_dim=32), got {actions.shape}")
    print("State:", tuple(obs.state.shape))
    if obs.state.shape[-1] != 32:
        raise ValueError("Expected state padded to 32")
    print("Images:", {key: tuple(val.shape) for key, val in obs.images.items()})
    print("Prompt tokens:", tuple(obs.tokenized_prompt.shape))
    print("OpenPI loader PASSED (no model weights or training executed)")


if __name__ == "__main__":
    main()
