# RECAP configuration plan — blocked until data-transform alignment

Reuse the upstream executable pipeline rather than copying its internals:

1. `examples/offline_rl/config/recap_compute_returns.yaml`: set SFT/rollout dataset paths, type, and a versioned `data.tag`.
2. `examples/offline_rl/config/recap_value_model_sft.yaml`: use same paths/returns tag, **verified** `robot_type`, `model_type=pi05`, image transform, critic assets (SigLIP2/Gemma3).
3. `examples/offline_rl/config/recap_compute_advantages.yaml`: same dataset paths, returns tag; output a distinct advantage tag; matching value checkpoint.
4. `examples/offline_rl/config/cfg_rl_openpi.yaml`: same positive/negative data and advantage tag; select *the same* OpenPI pi0.5 dataset transform and appropriate checkpoint.

**Important blocker:** `ValueDataset` maps `robot_type=franka` to `FrankaEEInputs` and the real-world SFT `pi05_franka_pnp` maps to `RealworldInputs`. The two disagree about image mapping, so templates that simply set `robot_type=franka` are not sufficient. Resolve with a *focused* upstream-compatible adapter after inspecting a real or representative LeRobot dataset. Do not run Value Model SFT/Advantage/CFG with an invented mapping.

**Only when data exist:** fix true episode success labels and failure policy; confirm action horizon/normalization; name checkpoint/asset directories; confirm RECAP tags are identical across stages; keep artifacts outside Git.

Acceptance gate: a single selected dataset sample must pass SFT OpenPI processing, RECAP value processing, advantage lookup, and CFG processing, with consistent image channels, prompt and action coordinates. This gate is **pending**.
