# Workstation storage policy — FR3 / OpenPI / RECAP

Large generated artifacts on GPU workstation `pca03395` belong under `/data1/Muttaqien/`. Source code remains in the Git repository.

| Artifact | Default location |
|---|---|
| FR3 raw demonstrations | `/data1/Muttaqien/fr3_peg_100/` |
| Converted datasets and RECAP rollouts | `/data1/Muttaqien/datasets/` |
| Downloaded/base model checkpoints | `/data1/Muttaqien/checkpoints/` |
| Training outputs and saved weights | `/data1/Muttaqien/training_runs/` or run-specific subdirectories of `logs/` |
| Launcher and TensorBoard logs | `/data1/Muttaqien/logs/` |
| Hugging Face, OpenPI, Transformers and Torch caches | `/data1/Muttaqien/cache/` |
| Temporary large files | `/data1/Muttaqien/tmp/` |

Launcher scripts source `scripts/rlinf_storage_env.sh` and use `RLINF_DATA_ROOT` (default `/data1/Muttaqien`). To choose a different mounted writable volume, set `RLINF_DATA_ROOT` before launching. Existing `HF_HOME`, `HF_DATASETS_CACHE`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `OPENPI_DATA_HOME`, `TORCH_HOME`, `WANDB_DIR`, `TMPDIR` environment overrides remain respected. The script does not move or delete existing files.

**Important:** The FR3 manual collector takes a required `--output-root` argument; it warns if that path is outside `/data1/Muttaqien/` but allows deliberate overrides. Before capturing, stop both ROS RealSense publishers so the collector can open the two devices directly. Keep the operator-approved FR3 controller, SpaceMouse and Robotiq bridge running.

**NUC exception:** `/home/aist-franka/fr3_telemetry_isolated_4c2t7/recordings/stationary_telemetry.jsonl` belongs to the separate NUC host. That file is *not* automatically copied or synchronized by the workstation collection tool. Preserve and transfer the NUC telemetry independently without restarting its recorder until verified.

The training configs are templates: configured dataset/checkpoint paths are **expected locations**, not claims that datasets, matching norm stats, pretrained assets, or policy registration already exist. Verify them before training. Upstream RLInf examples outside this FR3/RECAP workflow can retain other explicit paths; audit any new workflow separately.
