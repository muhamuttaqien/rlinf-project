"""Score paired LIBERO rollout observations with the trained SIM-4 value critic.

Reads synchronized agent-view + wrist-view MP4s and robot states recorded by
``recap_libero_sim7_eval --value-observations``. Predicts a *numerical* V(s_t)
with the same SIM-4 checkpoint used in SIM-5. This is NOT an advantage A(s,a),
and it is NOT an instantaneous success probability.

Example:
    python -m toolkits.recap_libero_value_curves \\
      --output logs/cfg_rl/recap_sim7_value_comparison_20episodes --episodes 20

No policy retraining or environment rollout is performed by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

DEFAULT_CHECKPOINT = Path(
    "/data1/Muttaqien/rlinf_data/recap_value_sft_full/"
    "libero10_task0_value_sft_full/checkpoints/global_step_8000/"
    "actor/model_state_dict/full_weights.pt"
)


def _load_video(path: Path) -> list[np.ndarray]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty value observation video: {path}")
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        frames = [np.asarray(f, dtype=np.uint8).copy() for f in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames in {path}")
    return frames


def _checkpoint_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        for candidate in (
            path / "actor/model_state_dict/full_weights.pt",
            path / "model_state_dict/full_weights.pt",
            path / "full_weights.pt",
        ):
            if candidate.is_file():
                return candidate
    if path.is_file() and path.suffix in (".pt", ".pth", ".safetensors"):
        return path
    raise FileNotFoundError(f"Value-model checkpoint not found: {path}")


def _load_value_critic(checkpoint: Path, device: str):
    """Use exactly the SIM-5 model configuration and loader, not an ad-hoc model."""
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.value_model.recap.modeling_critic import ValueCriticModel

    config_dir = Path(__file__).resolve().parents[1] / "examples/offline_rl/config"
    model_config = OmegaConf.load(config_dir / "model/recap_value_model.yaml")
    sim5_config = OmegaConf.load(config_dir / "recap_compute_advantages.yaml")
    model_cfg = OmegaConf.merge(model_config, sim5_config.advantage.model)
    if sim5_config.data.model_type != "pi05":
        raise ValueError("Unexpected SIM-5 value model type: expected pi05")
    if not str(checkpoint).endswith("full_weights.pt") and not str(checkpoint).endswith(".safetensors"):
        raise ValueError("Expected a full model-weights checkpoint, not a DCP optimizer shard")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run SIM-4 scoring on the lab GPU")

    print(f"Loading SIM-4 value critic: {checkpoint}", flush=True)
    model = ValueCriticModel.from_checkpoint(
        str(checkpoint), device=device,
        env_type="libero", model_type="pi05",
        num_return_bins=int(model_cfg.get("num_bins", 201)),
        return_min=float(model_cfg.get("v_min", -1.0)),
        return_max=float(model_cfg.get("v_max", 0.0)),
        critic_expert_variant=str(model_cfg.get("critic_expert_variant", "gemma_1m")),
        tokenizer_path=str(model_cfg.tokenizer_path),
        siglip_path=str(model_cfg.siglip_path),
        gemma3_path=str(model_cfg.gemma3_path),
    )
    return model


def _episode_observations(front_path: Path, episode: int, seed: int):
    """Require *both* camera streams and the recorded state for each frame."""
    wrist_path = front_path.with_name("wrist_" + front_path.name)
    state_path = front_path.parent / f"state_task0_episode{episode}.npz"
    if not wrist_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(
            f"Value observations unavailable for episode {episode}. "
            f"Expected {wrist_path} and {state_path}. The older single-camera videos "
            "cannot produce a comparable two-camera value curve. Rerun --prepare "
            "with --value-observations in a NEW --output directory."
        )
    front = _load_video(front_path)
    wrist = _load_video(wrist_path)
    with np.load(state_path, allow_pickle=False) as f:
        states = np.asarray(f["states"], dtype=np.float32)
        prompt = str(f["prompt"].item())
        recorded_seed = int(f["seed"])
        recorded_episode = int(f["episode"])
    if recorded_seed != seed or recorded_episode != episode:
        raise ValueError(f"State metadata does not match episode/seed at {state_path}")
    if len(front) != len(wrist) or len(front) != len(states):
        raise ValueError(
            f"Unsynchronized front/wrist/state frames for episode {episode}: "
            f"{len(front)}/{len(wrist)}/{len(states)}"
        )
    if states.ndim != 2 or states.shape[1] != 8 or not np.isfinite(states).all():
        raise ValueError(f"Expected finite (frames,8) LIBERO states in {state_path}")
    if not prompt:
        raise ValueError(f"Missing task prompt in {state_path}")
    observations = [
        {
            "observation/image": front[i],
            "observation/wrist_image": wrist[i],
            "observation/state": states[i],
            "prompt": prompt,
        }
        for i in range(len(front))
    ]
    return observations


def _write_curve(path: Path, episode: int, seed: int, policy: str, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("episode", "seed", "policy", "frame_index", "sim_step", "value"))
        writer.writeheader()
        for i, v in enumerate(values):
            v = float(v)
            if not math.isfinite(v):
                raise ValueError(f"Value model returned non-finite V(s) at episode {episode}, frame {i}")
            writer.writerow({
                "episode": episode, "seed": seed, "policy": policy,
                "frame_index": i, "sim_step": 10 + i * 10, "value": f"{v:.9g}",
            })
    pending.replace(path)


def score(output: Path, checkpoint: Path, episodes: int, seed: int, batch_size: int = 4, device: str = "cuda") -> None:
    output = output.expanduser().resolve()
    results_path = output / "paired_results.csv"
    manifest_path = output / "run_config.json"
    if not results_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing paired rollouts in {output}; run dashboard --prepare first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("episodes") != episodes or manifest.get("seed") != seed:
        raise ValueError("--episodes/--seed differ from recorded rollout manifest")
    if not manifest.get("value_observations", False):
        raise ValueError(
            "This run was recorded without two-camera value observations. "
            "Use dashboard --prepare with a new --output directory."
        )
    with results_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != episodes:
        raise ValueError(f"Expected {episodes} paired episodes, got {len(rows)}")

    ckpt = _checkpoint_file(checkpoint)
    values_dir = output / "values"
    values_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = values_dir / "metadata.json"
    metadata = {"checkpoint": str(ckpt), "checkpoint_size": ckpt.stat().st_size,
                "episodes": episodes, "seed": seed,
                "method": "SIM-4 ValueCriticModel.infer_batch on synchronized main/wrist/state LIBERO eval observations"}
    if metadata_path.is_file():
        prev = json.loads(metadata_path.read_text(encoding="utf-8"))
        if prev != metadata:
            raise ValueError("Existing value curves use different metadata/checkpoint; use a new --output directory")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pending = []
    for expected_ep, row in enumerate(rows, 1):
        ep, row_seed = int(row["episode"]), int(row["seed"])
        if ep != expected_ep or row_seed != seed + expected_ep - 1:
            raise ValueError(f"Unexpected paired-episode order/seed at episode {expected_ep}")
        for policy in ("baseline", "recap"):
            path = values_dir / f"{policy}_episode{ep:02d}.csv"
            if not path.is_file():
                rel = Path(row[f"{policy}_video"])
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"Unsafe MP4 path: {rel}")
                video = (output / rel).resolve()
                _episode_observations(video, ep, row_seed)  # preflight BEFORE loading multi-GB model
                pending.append((ep, row_seed, policy, path, video))
    if not pending:
        print("All SIM-4 value curves already exist; no GPU scoring needed.", flush=True)
        return

    model = _load_value_critic(ckpt, device)
    try:
        for index, (ep, row_seed, policy, path, video) in enumerate(pending, 1):
            print(f"Value scoring {index}/{len(pending)}: episode {ep} {policy}", flush=True)
            observations = _episode_observations(video, ep, row_seed)
            estimates = model.infer_batch(observations, batch_size=batch_size)
            if len(estimates) != len(observations):
                raise RuntimeError("Value-model output length differs from recorded video frames")
            _write_curve(path, ep, row_seed, policy, [x["value"] for x in estimates])
    finally:
        del model
    print(f"Saved SIM-4 value curves in {values_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/cfg_rl/recap_sim7_value_comparison_20episodes"))
    parser.add_argument("--value-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.episodes < 1 or args.batch_size < 1:
        parser.error("--episodes and --batch-size must be positive")
    score(args.output, args.value_checkpoint, args.episodes, args.seed, args.batch_size, args.device)


if __name__ == "__main__":
    main()
