# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Build StackCube TFDS/RLDS episodes for OpenVLA-OFT's RLDSDataset.

Install tensorflow-datasets and tensorflow in an isolated policy environment.
Use --dry-run for a NumPy-only schema/alignment check. This exporter records
raw ManiSkill pd_ee_delta_pose actions; normalization is NOT yet validated.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def episode_steps(source, entry, instruction):
    """Yield full RLDS-like steps with synchronized observations and actions."""
    with np.load(source / entry["path"], allow_pickle=False) as d:
        images, wrist, actions = d["base_rgb"], d["wrist_rgb"], d["actions"]
        if (images.ndim != 4 or images.shape[1:] != (256, 256, 3)
                or wrist.shape != images.shape or images.dtype != np.uint8
                or wrist.dtype != np.uint8 or actions.shape != (len(images)-1, 7)
                or not np.isfinite(actions).all()):
            raise ValueError(f"{entry['path']}: invalid episode")
        for t in range(len(actions)):
            yield {
                "observation": {"image": images[t], "wrist_image": wrist[t]},
                "action": actions[t].astype(np.float32),
                "language_instruction": instruction,
                "is_first": t == 0,
                "is_last": t == len(actions)-1,
                "is_terminal": t == len(actions)-1,
            }


def select_episodes(index_path, split):
    index = json.loads(Path(index_path).read_text())
    source = Path(index["source_dir"])
    selected = index["episodes"][split]
    catalog = {x["episode"]: x["file"] for x in index["splits"][split]}
    return source, [{"episode": x, "path": catalog[x]} for x in selected]


def dry_run(index_path):
    counts = {}
    for split in ("train", "val"):
        source, entries = select_episodes(index_path, split)
        total = 0
        for entry in tqdm(entries, desc=f"Checking {split}", unit="episode"):
            total += sum(1 for _ in episode_steps(
                source, entry, "Stack the red cube on top of the green cube."))
        counts[split] = {"episodes": len(entries), "steps": total}
    return counts


def build(index_path, output, dataset_name="stackcube_panda"):
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError("RLDS export requires tensorflow and tensorflow-datasets") from exc

    index_path = Path(index_path)
    class StackCubeBuilder(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "ManiSkill StackCube Panda dual-camera actions"}

        def _info(self):
            return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict({
                    "steps": tfds.features.Dataset({
                        "observation": tfds.features.FeaturesDict({
                            "image": tfds.features.Image(shape=(256, 256, 3)),
                            "wrist_image": tfds.features.Image(shape=(256, 256, 3)),
                        }),
                        "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                        "language_instruction": tfds.features.Text(),
                        "is_first": tf.bool,
                        "is_last": tf.bool,
                        "is_terminal": tf.bool,
                    })
                })
            )

        def _split_generators(self, dl_manager):
            return {"train": self._generate_examples("train"),
                    "validation": self._generate_examples("val")}

        def _generate_examples(self, split):
            source, entries = select_episodes(index_path, split)
            for entry in tqdm(entries, desc=f"Writing {split}", unit="episode"):
                yield entry["episode"], {
                    "steps": list(episode_steps(
                        source, entry, "Stack the red cube on top of the green cube."))
                }

    # Dataset name is derived from the class by TFDS; use the standard
    # stack_cube_builder name unless explicitly registered in upstream.
    builder = StackCubeBuilder(data_dir=str(output))
    builder.download_and_prepare()
    return str(builder.data_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path,
                        default=Path("data/stackcube_openvlaoft_sft/index.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/stackcube_rlds"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run(args.index), indent=2))
    else:
        print("TFDS dataset written:", build(args.index, args.output))


if __name__ == "__main__":
    main()
