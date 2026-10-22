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

"""One-frame native OpenPI train/eval input and action-transform parity gate.

Run after new-data collection and norm-stats calculation but BEFORE SFT.
This reads one raw LeRobot sample, runs actual OpenPI transformations,
and checks train/eval equality plus normalization/padding round-trip.
It does not load the 3.35B model or modify files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rlinf.data.datasets.openpi.v3_stackcube_dataset import (
    V3StackCubeDataset, is_lerobot_v3_dataset,
)
from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
from toolkits.lerobot.stackcube_ee_delta import TASK


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def camera_to_hwc_uint8(value):
    """Match ManiSkillInputs' float->uint8 conversion of a LeRobot image."""
    x = as_numpy(value)
    if x.shape == (3, 224, 224):
        x = np.transpose(x, (1, 2, 0))
    if x.shape != (224, 224, 3):
        raise ValueError(f"Unexpected camera shape: {x.shape}")
    if np.issubdtype(x.dtype, np.floating):
        x = (255 * x).astype(np.uint8)
    if x.dtype != np.uint8:
        raise ValueError(f"Unexpected RGB dtype: {x.dtype}")
    return x


def raw_sample(repo_id: str):
    root = resolve_lerobot_dataset_root(repo_id)
    if is_lerobot_v3_dataset(root):
        reader = V3StackCubeDataset(
            root, action_horizon=10, state_dim=8, action_dim=7,
            use_wrist_image=True,
        )
    else:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Same future actions and frame spacing that OpenPI requests.
        import json

        fps = float(
            json.loads((root / "meta" / "info.json").read_text())["fps"]
        )
        reader = LeRobotDataset(
            repo_id=repo_id, root=root,
            delta_timestamps={"actions": [t / fps for t in range(10)]},
        )
    return reader[0]


def verify_transform_roundtrip(
    repo_id: str, norm_stats_path: Path, model_path: Path,
) -> None:
    from openpi import transforms
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.transforms.pipeline import (
        build_openpi_transforms,
    )

    if not norm_stats_path.is_file():
        raise FileNotFoundError(f"Fresh normalization stats missing: {norm_stats_path}")
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Original π0.5 base weights missing: {model_path}")

    sample = raw_sample(repo_id)
    state = as_numpy(sample["state"]).astype(np.float32)
    expected_actions = as_numpy(sample["actions"]).astype(np.float32)
    if state.shape != (8,) or expected_actions.shape != (10, 7):
        raise ValueError(
            f"Raw sample must be 8D state and [10,7] actions, got "
            f"{state.shape} and {expected_actions.shape}"
        )

    cfg_name = "pi05_stackcube_panda_easy_ee_delta_two_cam_state"
    kwargs = {"norm_stats_path": str(norm_stats_path.resolve())}
    train_cfg = get_openpi_config(
        cfg_name, model_path=str(model_path.resolve()),
        repo_id=repo_id, data_kwargs=kwargs,
    )
    data_cfg = train_cfg.data.create(
        train_cfg.assets_dirs, train_cfg.model,
    )
    if data_cfg.norm_stats is None:
        raise ValueError("OpenPI training config did not load fresh norm_stats")
    inp, out = build_openpi_transforms(str(model_path), cfg_name, kwargs)
    repack = transforms.compose(data_cfg.repack_transforms.inputs)
    input_fn, output_fn = transforms.compose(inp), transforms.compose(out)

    # Training's raw LeRobot sample goes through Repack + OpenPI input.
    train_values = input_fn(repack(sample))
    # Inference's raw sensor/state values go through the SAME OpenPI input.
    eval_values = input_fn({
        "observation/image": camera_to_hwc_uint8(sample["image"]),
        "observation/wrist_image": camera_to_hwc_uint8(sample["wrist_image"]),
        "observation/state": state,
        "prompt": TASK,
    })

    for key in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
        np.testing.assert_allclose(
            as_numpy(train_values[key]),
            as_numpy(eval_values[key]),
            atol=1e-6, rtol=0,
            err_msg=f"Train/eval OpenPI {key} mismatch",
        )
    for key in ("base_0_rgb", "left_wrist_0_rgb"):
        np.testing.assert_array_equal(
            as_numpy(train_values["image"][key]),
            as_numpy(eval_values["image"][key]),
            err_msg=f"Train/eval image mismatch: {key}",
        )
        if not bool(np.asarray(train_values["image_mask"][key]).item()):
            raise AssertionError(f"{key} is masked off in SFT")
    if bool(np.asarray(train_values["image_mask"]["right_wrist_0_rgb"]).item()):
        raise AssertionError("Placeholder third camera must be masked")

    model_actions = as_numpy(train_values["actions"])
    if model_actions.shape != (10, 32):
        raise AssertionError(
            f"OpenPI must pad 7D actions to [10,32], got {model_actions.shape}"
        )
    decoded = output_fn({
        "actions": model_actions,
        "state": as_numpy(train_values["state"]),
    })["actions"]
    np.testing.assert_allclose(
        as_numpy(decoded), expected_actions, rtol=1e-4, atol=1e-4,
        err_msg="OpenPI normalization+padding+decoding changed 7D action targets",
    )
    print(
        "OPENPI TRANSFORMS PASSED "
        "train/eval images, state, tokens identical; "
        "7D -> normalized 32D -> decoded 7D action round-trip"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id", default="local/stackcube_pi05_clean_v1",
    )
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    args = parser.parse_args()
    verify_transform_roundtrip(
        args.repo_id,
        args.norm_stats.expanduser().resolve(),
        args.model_path.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
