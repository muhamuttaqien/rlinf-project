# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import numpy as np
import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
import tqdm
import tyro
from openpi.training.config import DataConfig

from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def create_torch_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.TorchDataLoader, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def compute_v3_lerobot_norm_stats(
    dataset_root: Path,
) -> dict[str, normalize.NormStats]:
    """Stream numerical columns from a LeRobot v3 dataset without v2 metadata."""
    import pyarrow.parquet as pq

    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    dimensions = {}
    for name in ("state", "actions"):
        feature = features.get(name)
        if not feature or len(feature.get("shape", [])) != 1:
            raise ValueError(f"Expected a 1D {name} feature in {info_path}")
        dimensions[name] = int(feature["shape"][0])

    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No data Parquet files under {dataset_root / 'data'}")

    stats = {key: normalize.RunningStats() for key in dimensions}
    seen = 0
    for file in tqdm.tqdm(parquet_files, desc="Computing LeRobot v3 stats"):
        parquet = pq.ParquetFile(file)
        missing = set(dimensions) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{file} missing columns {sorted(missing)}")
        for batch in parquet.iter_batches(
            batch_size=1024, columns=list(dimensions)
        ):
            for key, dim in dimensions.items():
                values = np.asarray(batch.column(key).to_pylist(), dtype=np.float32)
                if values.ndim != 2 or values.shape[1] != dim:
                    raise ValueError(
                        f"{file}: expected {key} shape [N, {dim}], got {values.shape}"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(f"{file}: {key} contains NaN or infinity")
                stats[key].update(values)
            seen += batch.num_rows

    expected = info.get("total_frames")
    if expected is not None and seen != int(expected):
        raise ValueError(
            f"LeRobot v3 frame count mismatch: info.json says {expected}, "
            f"but data Parquet files contain {seen} rows"
        )
    if not seen:
        raise ValueError(f"No frames found under {dataset_root / 'data'}")
    return {key: value.get_statistics() for key, value in stats.items()}


def create_rlds_dataloader(
    data_config: DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(
        data_config, action_horizon, batch_size, shuffle=False
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    repo_id: str,
    output_root: str | None = None,
):
    dataset_root = resolve_lerobot_dataset_root(repo_id)
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"LeRobot dataset not found for repo_id={repo_id!r} at {dataset_root}. "
            "Pass a local dataset path, a Hugging Face repo id with data under "
            "HF_LEROBOT_HOME (default: ~/.cache/huggingface/lerobot), or download "
            "the dataset first."
        )
    config = get_openpi_config(
        config_name,
        repo_id=repo_id,
    )

    # OpenPI's pinned LeRobot loader expects v2 JSONL metadata. A v3 dataset
    # stores multiple episodes per Parquet shard with task metadata in Parquet,
    # so compute the same raw state/action statistics directly from its shards.
    if (
        (dataset_root / "meta" / "tasks.parquet").is_file()
        and not (dataset_root / "meta" / "tasks.jsonl").is_file()
    ):
        norm_stats = compute_v3_lerobot_norm_stats(dataset_root)
        output_path = (
            Path(output_root).expanduser() / repo_id
            if output_root else config.assets_dirs / repo_id
        )
        print(f"Writing LeRobot v3 stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return

    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = (
        Path(output_root).expanduser() / data_config.repo_id
        if output_root else config.assets_dirs / data_config.repo_id
    )
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
