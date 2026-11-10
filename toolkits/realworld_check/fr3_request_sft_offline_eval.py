# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Read-only checkpoint reload and held-out π0.5 FR3 request-action evaluation.

Uses RLinf's exact Pi0 full_weights.pt loader and the official, strict-horizon
OpenPI validation transforms. No ROS, robot commands, training or RECAP.
Predictions are normalized SpaceMouse REQUESTS, not accepted robot actions.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from toolkits.realworld_check.fr3_request_openpi_preflight import validate_metadata

ROOT = Path(__file__).resolve().parents[2]
CONFIG_NAME = "pi05_fr3_peg_request_offline"
FIELDS = ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z")


def select_windows(segments: list[dict], *, horizon: int = 10,
                   per_episode: int = 5, all_windows: bool = False
                   ) -> tuple[list[tuple[int, str]], int]:
    """Return deterministic positions in the official strict-window dataset.

    The exporter calls save_episode once per segment, in manifest order.
    Thus each segment of length N contributes N-horizon+1 consecutive
    strict-window dataset indices, with no crossing of episode boundaries.
    """
    if horizon < 1 or per_episode < 1 or not segments:
        raise ValueError("Invalid horizon, per_episode or segments")
    per_source: dict[str, list[int]] = OrderedDict()
    cursor = 0
    for segment in segments:
        source = segment.get("episode")
        first, stop = segment.get("start"), segment.get("stop")
        if (not isinstance(source, str) or not source.startswith("episode_")
                or type(first) is not int or type(stop) is not int
                or stop - first < horizon):
            raise ValueError("Invalid LeRobot source segment metadata")
        count = stop - first - horizon + 1
        per_source.setdefault(source, []).extend(range(cursor, cursor + count))
        cursor += count
    selected = []
    for source, indices in per_source.items():
        if all_windows:
            chosen = indices
        else:
            sample_positions = np.linspace(
                0, len(indices) - 1, num=min(per_episode, len(indices)), dtype=int
            )
            chosen = [indices[int(i)] for i in sample_positions]
        selected.extend((index, source) for index in chosen)
    selected.sort(key=lambda pair: pair[0])
    if not selected:
        raise ValueError("No validation windows selected")
    return selected, cursor


def metric_summary(predicted: np.ndarray, true: np.ndarray) -> dict:
    """Metrics in ORIGINAL normalized SpaceMouse request units (not OpenPI units)."""
    pred, truth = np.asarray(predicted, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if (pred.shape != truth.shape or pred.ndim != 3
            or pred.shape[1:] != (10, 7) or not np.isfinite(pred).all()
            or not np.isfinite(truth).all()):
        raise ValueError("Expected finite matched (windows,10,7) request arrays")
    if (np.max(np.abs(truth[..., :6])) > 1.002
            or not np.allclose(truth[..., 6], np.round(truth[..., 6]), atol=2e-3)
            or not np.isin(np.round(truth[..., 6]), [0, 1]).all()):
        raise ValueError("Decoded validation targets are not raw 6D requests + binary gripper")
    axes = {}
    for i, name in enumerate(FIELDS):
        p, y = pred[..., i].ravel(), truth[..., i].ravel()
        error = p - y
        r = (float(np.corrcoef(p, y)[0, 1])
             if np.std(p) > 1e-8 and np.std(y) > 1e-8 else None)
        axes[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "zero_request_baseline_rmse": float(np.sqrt(np.mean(y ** 2))),
            "pearson_r": r,
        }
    y = truth[..., 6] >= 0.5
    p = pred[..., 6] >= 0.5
    tp, tn = int(np.sum(p & y)), int(np.sum(~p & ~y))
    fp, fn = int(np.sum(p & ~y)), int(np.sum(~p & y))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return {
        "action_steps_counted": int(pred.shape[0] * pred.shape[1]),
        "arm": axes,
        "arm_6d_mae": float(np.mean(np.abs(pred[..., :6] - truth[..., :6]))),
        "arm_6d_rmse": float(np.sqrt(np.mean((pred[..., :6] - truth[..., :6]) ** 2))),
        "predicted_arm_outside_request_range_fraction": float(
            np.mean(np.abs(pred[..., :6]) > 1.0)
        ),
        "gripper_open": {
            "threshold": 0.5,
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "majority_class_baseline_accuracy": max(tp + fn, tn + fp) / (tp + tn + fp + fn),
            "precision": precision, "recall": recall, "f1": f1,
            "true_open": tp + fn, "true_close": tn + fp,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "continuous_mae": float(np.mean(np.abs(pred[..., 6] - truth[..., 6]))),
        },
    }


def make_model_config(ckpt: Path):
    from omegaconf import OmegaConf
    template = OmegaConf.load(ROOT / "examples/sft/config/model/pi0_5.yaml")
    experiment = OmegaConf.load(
        ROOT / "examples/sft/config/realworld_sft_openpi_fr3_peg_request_full3000_wandb.yaml"
    )
    cfg = OmegaConf.merge(template, experiment.actor.model)
    if (cfg.action_dim != 7 or cfg.openpi.config_name != CONFIG_NAME
            or cfg.openpi.model_action_dim != 32 or cfg.openpi.action_horizon != 10
            or cfg.openpi.task != "sft"):
        raise ValueError("Checkpoint/model config is not the trained 7D request SFT shape")
    cfg.model_path = str(ckpt)
    cfg.openpi.torch_compile = False
    return cfg


def evaluate(args) -> dict:
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available; use the NVIDIA 590 library override")

    ckpt = args.checkpoint.expanduser().resolve()
    if ckpt.name != "full_weights.pt" or not ckpt.is_file() or ckpt.stat().st_size < 1_000_000_000:
        raise ValueError("Pass the nonempty model_state_dict/full_weights.pt checkpoint file")
    if "global_step_3000" not in ckpt.parts:
        raise ValueError("Expected fresh global_step_3000 SFT checkpoint")
    if not args.reload_only:
        provenance = validate_metadata(args.train, args.validation, args.norm_dir)
    else:
        provenance = None

    from rlinf.models.embodiment.openpi import get_model

    cfg = make_model_config(ckpt)
    print("Loading consolidated π0.5 checkpoint with RLinf model factory...", flush=True)
    model = get_model(cfg)
    model.eval().to(args.device)
    params = sum(p.numel() for p in model.parameters())
    if not 3_000_000_000 < params < 4_000_000_000:
        raise ValueError(f"Unexpected parameter count: {params}")
    first = next(model.parameters())
    if not torch.isfinite(first.detach()).all():
        raise ValueError("Nonfinite parameter detected in loaded model")
    result = {
        "schema": "fr3_request_sft_offline_heldout_eval_v1",
        "checkpoint": str(ckpt),
        "checkpoint_reload_passed": True,
        "parameter_count": params,
        "model": "pi05_fr3_peg_request_offline",
        "device": args.device,
        "physical_robot_execution_approved": False,
        "retraining": False,
    }
    print("Checkpoint reload PASSED", flush=True)
    if args.reload_only:
        return result

    from omegaconf import OmegaConf
    from openpi import transforms
    from openpi.training import data_loader as pi_loader
    from rlinf.data.datasets.openpi.official_sft_data_loader import (
        build_official_openpi_sft_dataloader,
    )

    loader_cfg = OmegaConf.create({
        "data": {"num_workers": 0},
        "actor": {
            "seed": 0, "micro_batch_size": 1, "eval_batch_size": 1,
            "model": {
                "model_path": str(args.base_model),  # Keep original OpenPI tokenizer/assets.
                "openpi": {"config_name": CONFIG_NAME, "model_action_dim": 32,
                           "action_horizon": 10},
                "openpi_data": {
                    "norm_stats_path": str(args.norm_dir / "norm_stats.json")
                },
            },
        },
    })
    loader, data_cfg = build_official_openpi_sft_dataloader(
        loader_cfg, world_size=1, rank=0,
        data_paths=str(args.validation), eval_dataset=True,
    )
    manifest = json.loads(
        (args.validation / "fr3_command_request_provenance.json").read_text(encoding="utf-8")
    )
    selected, total = select_windows(
        manifest["segments"], per_episode=args.per_episode, all_windows=args.all_windows
    )
    if (total != loader.strict_valid_starts_count
            or total != 2898 or len({ep for _, ep in selected}) != 20):
        raise ValueError("Held-out 20-episode strict-window provenance is inconsistent")
    # Same transformed samples and collator as the official OpenPI loader,
    # but restrict selected indices so each of the 20 episodes contributes.
    original = loader._data_loader._data_loader
    transformed = original.dataset
    if len(transformed) != total:
        raise ValueError("Validation loader index layout changed")
    subset = torch.utils.data.Subset(transformed, [i for i, _ in selected])
    subset_loader = pi_loader.DataLoaderImpl(
        data_cfg,
        pi_loader.TorchDataLoader(
            subset, local_batch_size=1, shuffle=False, num_workers=0,
            seed=0, framework="pytorch", num_batches=len(selected),
        ),
    )
    if data_cfg.norm_stats is None or "actions" not in data_cfg.norm_stats:
        raise ValueError("Missing train-only official action stats")
    inverse = transforms.Unnormalize(
        {"actions": data_cfg.norm_stats["actions"]},
        use_quantiles=data_cfg.use_quantile_norm,
    )

    device = torch.device(args.device)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    predicted = []
    targets = []
    sources = []
    with torch.inference_mode():
        for k, ((obs, actions), (_, episode)) in enumerate(
                zip(subset_loader, selected, strict=True), 1):
            obs = model._observation_to_device(obs)
            noise = torch.randn(
                (1, 10, 32), device=device, generator=rng, dtype=torch.float32
            )
            generated = model.sample_actions(
                obs, num_steps=args.denoising_steps, noise=noise
            )
            p = generated.detach().cpu().float().numpy()[0]
            y = actions.detach().cpu().float().numpy()[0]
            if p.shape != (10, 32) or y.shape != (10, 32):
                raise ValueError("Invalid model output or official action dimensions")
            # Inverse official OpenPI normalization, then retain only 7D
            # request labels. There is no executable action adapter here.
            p = np.asarray(inverse({"actions": p})["actions"][:, :7])
            y = np.asarray(inverse({"actions": y})["actions"][:, :7])
            if not np.isfinite(p).all() or not np.isfinite(y).all():
                raise ValueError("Nonfinite model predictions / targets")
            predicted.append(p)
            targets.append(y)
            sources.append(episode)
            if k % 20 == 0:
                print(f"Evaluated {k}/{len(selected)} offline validation windows", flush=True)
    pred, target = np.stack(predicted), np.stack(targets)
    if getattr(args, "transition_output", None) is not None:
        if not args.all_windows:
            raise ValueError("--transition-output requires --all-windows")
        from toolkits.realworld_check.fr3_request_transition_diagnostic import analyze
        diagnostic = analyze(
            pred, target, manifest["segments"],
            threshold=args.activity_threshold,
            tolerance=args.event_tolerance,
        )
        target_path = args.transition_output.expanduser()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("x", encoding="utf-8") as output_file:
            json.dump(diagnostic, output_file, indent=2, allow_nan=False)
            output_file.write("\\n")
        print(f"Transition report saved: {target_path}", flush=True)

    result.update({
        "source_episode_disjoint": provenance["source_episode_disjoint"],
        "train_only_norm_passed": provenance["train_only_norm_passed"],
        "heldout_source_episodes": len(set(sources)),
        "heldout_possible_windows": total,
        "evaluated_windows": len(sources),
        "window_selection": "all_windows" if args.all_windows else "evenly_spaced_per_episode",
        "per_episode_requested": None if args.all_windows else args.per_episode,
        "denoising_steps": args.denoising_steps,
        "seed": args.seed,
        "metric_units": "original normalized SpaceMouse requests, NOT TCP SI units",
        "overlapping_action_horizons": True,
        "metrics": metric_summary(pred, target),
        "per_episode": {
            episode: metric_summary(
                pred[np.asarray(sources) == episode],
                target[np.asarray(sources) == episode],
            ) for episode in dict.fromkeys(sources)
        },
        "caveats": [
            "Offline teacher-forced observation windows, not closed-loop peg insertion.",
            "Overlapping 10-step action horizons are correlated, not independent trajectories.",
            "No calibrated controller acceptance, camera exposure timing or robot safety.",
            "Euler flow-model outputs depend on inference noise seed and denoising steps.",
        ],
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Exact final .../actor/model_state_dict/full_weights.pt")
    parser.add_argument("--base-model", type=Path, default=Path(
        "/data1/Muttaqien/rlinf_data/checkpoints/torch/pi05_base_rlinf"
    ))
    parser.add_argument("--train", type=Path, default=Path(
        "/data1/Muttaqien/fr3_peg_request_lerobot_train_v1"
    ))
    parser.add_argument("--validation", type=Path, default=Path(
        "/data1/Muttaqien/fr3_peg_request_lerobot_validation_v1"
    ))
    parser.add_argument("--norm-dir", type=Path, default=Path(
        "/data1/Muttaqien/fr3_peg_request_norm_v1"
    ))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--reload-only", action="store_true")
    parser.add_argument("--per-episode", type=int, default=5)
    parser.add_argument("--all-windows", action="store_true",
                        help="Evaluate all 2898 windows; significantly slower")
    parser.add_argument("--denoising-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transition-output", type=Path,
                        help="Write offline first-step transition report (requires --all-windows)")
    parser.add_argument("--activity-threshold", type=float, default=0.1)
    parser.add_argument("--event-tolerance", type=int, default=2,
                        help="Maximum nominal frame difference for event matching")
    parser.add_argument("--output", type=Path, help="New JSON result; never overwrite")
    args = parser.parse_args()
    if not 1 <= args.denoising_steps <= 20 or args.per_episode < 1:
        parser.error("denoising_steps must be 1..20 and per_episode >=1")
    if args.transition_output is not None and not args.all_windows:
        parser.error("--transition-output requires --all-windows")
    if not 0 < args.activity_threshold < 1 or not 0 <= args.event_tolerance <= 10:
        parser.error("activity threshold must be in (0,1), tolerance in [0,10]")
    report = evaluate(args)
    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as dst:
            json.dump(report, dst, indent=2, allow_nan=False)
            dst.write("\n")
        print(f"Report saved: {output}", flush=True)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
