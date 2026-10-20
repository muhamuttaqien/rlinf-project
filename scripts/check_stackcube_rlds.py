# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""Read one StackCube TFDS/RLDS episode from the exported dataset."""
import argparse
from pathlib import Path


def check(dataset_dir):
    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(str(dataset_dir))
    print("dataset:", builder.name)
    for split in ("train", "validation"):
        episodes = builder.as_dataset(split=split, shuffle_files=False)
        episode = next(iter(tfds.as_numpy(episodes.take(1))))
        steps = episode["steps"]
        step = next(iter(steps))
        image = step["observation"]["image"]
        wrist = step["observation"]["wrist_image"]
        action = step["action"]
        assert image.shape == (256, 256, 3), image.shape
        assert wrist.shape == (256, 256, 3), wrist.shape
        assert action.shape == (7,), action.shape
        print(f"{split}: base={image.shape}, wrist={wrist.shape}, action={action.shape}")
    print("RLDS read check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path,
                        default=Path("data/stackcube_rlds/stack_cube_builder/1.0.0"))
    check(parser.parse_args().dataset_dir)
