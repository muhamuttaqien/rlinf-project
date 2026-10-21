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

"""Exercise StackCube pilot auditing and RECAP return preparation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_lerobot_dataset(
    tmp_path: Path, *, indexed_tasks: bool = False, all_successful: bool = False
) -> tuple[Path, Path]:
    root = tmp_path / "stackcube"
    (root / "meta").mkdir(parents=True)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    task = "stack the red cube on top of the green cube and release it"
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 2,
                "total_frames": 4,
                "features": {
                    "image": {"dtype": "video", "shape": [224, 224, 3]},
                    "wrist_image": {"dtype": "video", "shape": [224, 224, 3]},
                    "state": {"dtype": "float32", "shape": [9]},
                    "actions": {"dtype": "float32", "shape": [8]},
                    "is_success": {"dtype": "bool", "shape": [1]},
                },
            }
        ),
        encoding="utf-8",
    )
    task_text_column = "__index_level_0__" if indexed_tasks else "task"
    pq.write_table(
        pa.table({"task_index": [0], task_text_column: [task]}),
        root / "meta" / "tasks.parquet",
    )
    filepath = data_dir / "file-000.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 0, 1, 1],
                "frame_index": [0, 1, 0, 1],
                "task_index": [0, 0, 0, 0],
                "state": pa.array([[0.0] * 9] * 4, type=pa.list_(pa.float32(), 9)),
                "actions": pa.array([[0.0] * 8] * 4, type=pa.list_(pa.float32(), 8)),
                "is_success": pa.array(
                    [[True], [True], [all_successful], [all_successful]],
                    type=pa.list_(pa.bool_(), 1),
                ),
            }
        ),
        filepath,
    )
    return root, filepath


def test_pilot_audit_rejects_failure_in_sft(tmp_path):
    audit = _load_script(
        "stackcube_audit",
        "toolkits/lerobot/validate_maniskill_stackcube_recap.py",
    )
    root, _ = _small_lerobot_dataset(tmp_path)
    with pytest.raises(ValueError, match="SFT data must succeed"):
        audit.audit_dataset(root)


def test_pilot_audit_accepts_correct_sft_frames():
    audit = _load_script(
        "stackcube_audit",
        "toolkits/lerobot/validate_maniskill_stackcube_recap.py",
    )
    sample = {
        "state": [0.0] * 9,
        "actions": [0.0] * 8,
        "is_success": [True],
    }
    frames = {
        0: [{"frame_index": 0, **sample}, {"frame_index": 1, **sample}],
        1: [{"frame_index": 0, **sample}],
    }
    result = audit.validate_episode_rows(frames)
    assert result["episodes"] == 2
    assert result["frames"] == 3
    assert result["successful_episodes"] == 2
    assert audit.scalar_bool([False]) is False


def test_recap_stage1_handles_shape_one_failure_and_v3_task(tmp_path):
    returns_module = pytest.importorskip("hydra") and _load_script(
        "stackcube_returns",
        "examples/offline_rl/advantage_labeling/recap/process/compute_returns.py",
    )
    root, filename = _small_lerobot_dataset(tmp_path)
    tasks = returns_module._load_dataset_tasks(root)
    assert tasks == {0: "stack the red cube on top of the green cube and release it"}
    result = returns_module._process_single_parquet(
        str(filename),
        dataset_type="rollout",
        gamma=1.0,
        failure_reward=-300.0,
        tasks=tasks,
    )
    assert result.column("reward").to_pylist() == [-1.0, 0.0, -1.0, -300.0]
    assert result.column("return").to_pylist() == [-1.0, 0.0, -301.0, -300.0]
    assert all(prompt == tasks[0] for prompt in result.column("prompt").to_pylist())


def test_audit_detects_wrong_action_dimension():
    audit = _load_script(
        "stackcube_audit",
        "toolkits/lerobot/validate_maniskill_stackcube_recap.py",
    )
    with pytest.raises(ValueError, match="invalid action"):
        audit.validate_episode_rows(
            {0: [{"frame_index": 0, "state": [0.0] * 9,
                  "actions": [0.0] * 7, "is_success": [True]}]}
        )


@pytest.mark.parametrize("indexed_tasks", [False, True])
def test_task_metadata_layouts_validate_successful_episodes(tmp_path, indexed_tasks):
    """Audit SFT episodes with both explicit task and pandas-indexed task text."""
    audit = _load_script(
        "stackcube_audit",
        "toolkits/lerobot/validate_maniskill_stackcube_recap.py",
    )
    root, _ = _small_lerobot_dataset(
        tmp_path, indexed_tasks=indexed_tasks, all_successful=True
    )
    task_columns = pq.read_schema(root / "meta" / "tasks.parquet").names
    expected_column = "__index_level_0__" if indexed_tasks else "task"
    assert expected_column in task_columns
    report = audit.audit_dataset(root, expected_episodes=2)
    assert report["successful_episodes"] == 2
    assert report["frames"] == 4


def test_recap_stage1_reads_pandas_indexed_v3_task_and_rewards(tmp_path):
    """Regression for LeRobot 0.4 tasks.parquet with __index_level_0__."""
    pytest.importorskip("hydra")
    returns_module = _load_script(
        "stackcube_returns",
        "examples/offline_rl/advantage_labeling/recap/process/compute_returns.py",
    )
    root, filename = _small_lerobot_dataset(tmp_path, indexed_tasks=True)
    tasks = returns_module._load_dataset_tasks(root)
    assert tasks == {0: "stack the red cube on top of the green cube and release it"}

    rows = returns_module._process_single_parquet(
        str(filename),
        dataset_type="rollout",
        gamma=1.0,
        failure_reward=-300.0,
        tasks=tasks,
    )
    assert rows.column("prompt").to_pylist() == [tasks[0]] * 4
    assert rows.column("reward").to_pylist() == [-1.0, 0.0, -1.0, -300.0]
