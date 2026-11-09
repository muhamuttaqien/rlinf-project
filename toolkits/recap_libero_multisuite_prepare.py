"""Record one task (Task 0) and 20 matched SFT/RECAP episodes per LIBERO suite.

The existing LIBERO-10 Task 0 recordings are preserved. By default this
script prepares the four other suites and uses the SIM-4 critic for V(s).

Example:
    python -m toolkits.recap_libero_multisuite_prepare

Select suites or preview commands without running a GPU experiment:
    python -m toolkits.recap_libero_multisuite_prepare --suites libero_spatial --dry-run

All experiments are exploratory cross-suite evaluations of a policy fine-tuned
on LIBERO-10 Task 0, NOT a claim of retraining for the other benchmarks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_90")
DEFAULT_CHECKPOINT = Path(
    "logs/cfg_rl/experiments/libero10_task0/"
    "cfg_rl_openpi-20261109-06:59:55/recap_libero10_task0_cfg/"
    "checkpoints/global_step_3000"
)
DEFAULT_OUTPUT = Path("logs/cfg_rl/recap_sim7_value_comparison_20episodes")
DEFAULT_SFT = Path("/data1/Muttaqien/rlinf_data/checkpoints/torch/pi05_libero_sft_rlinf")


def output_for_suite(long_output: Path, suite: str) -> Path:
    if suite not in SUITES:
        raise ValueError(f"Unknown LIBERO suite: {suite}")
    return long_output if suite == "libero_10" else long_output.with_name(long_output.name + "_" + suite)


def run(args: argparse.Namespace) -> None:
    from toolkits.recap_libero_web_dashboard import load_experiment

    root = Path(__file__).resolve().parents[1]
    long_output = args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    if not args.dry_run:
        weights = checkpoint / "actor/model_state_dict/full_weights.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"RECAP checkpoint not found: {weights}")
        if not model_path.is_dir():
            raise FileNotFoundError(f"SFT model not found: {model_path}")

    for suite in args.suites:
        output = output_for_suite(long_output, suite)
        ready = False
        if not args.dry_run and (output / "paired_results.csv").exists():
            # A file existing is not enough: validate episode, policy, seed,
            # and exact suite identity to prevent mislabeled evaluation.
            load_experiment(
                output, expected_episodes=args.episodes,
                expected_seed=args.seed, expected_suite=suite, expected_task_index=0,
            )
            ready = True

        if ready:
            print(f"Reuse verified {suite} Task 0: {args.episodes} paired episodes", flush=True)
        else:
            cmd = [
                sys.executable, "-m", "toolkits.recap_libero_compare_videos",
                "--checkpoint", str(checkpoint),
                "--model-path", str(model_path),
                "--suite", suite, "--task-index", "0",
                "--episodes", str(args.episodes), "--seed", str(args.seed),
                "--output", str(output),
                "--value-observations", "--resume",
            ]
            print(f"\nPrepare {suite} Task 0: {args.episodes} paired episodes", flush=True)
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=root, check=True)

        if not args.skip_values:
            cmd = [
                sys.executable, "-m", "toolkits.recap_libero_value_curves",
                "--output", str(output),
                "--episodes", str(args.episodes), "--seed", str(args.seed),
                "--batch-size", str(args.value_batch_size),
            ]
            print(f"Compute/reuse SIM-4 V(s) for {suite}", flush=True)
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=root, check=True)

    print("\nSuite preparation complete. Restart the browser dashboard to refresh available categories.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="LIBERO-10 Task 0 output; suite folders are siblings")
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(DEFAULT_SUITES))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--value-batch-size", type=int, default=4)
    parser.add_argument("--skip-values", action="store_true", help="Skip SIM-4 value scoring")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing evaluations")
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or args.value_batch_size < 1:
        parser.error("--episodes and --value-batch-size must be >=1; seed must be >=0")
    if len(set(args.suites)) != len(args.suites):
        parser.error("--suites must be unique")
    run(args)


if __name__ == "__main__":
    main()
