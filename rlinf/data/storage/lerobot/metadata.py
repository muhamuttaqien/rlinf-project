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

"""Read task descriptions from LeRobot v2 and v3 metadata formats."""

from __future__ import annotations

import json
from pathlib import Path


def load_task_descriptions(dataset_path: str | Path) -> dict[int, str]:
    """Map task indices to prompts in LeRobot JSONL or Parquet metadata.

    LeRobot v3 can save the task text as a pandas index, written physically
    as the Parquet column ``__index_level_0__`` rather than ``task``.
    """
    meta = Path(dataset_path) / "meta"
    tasks: dict[int, str] = {}

    jsonl_path = meta / "tasks.jsonl"
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                index = int(record.get("task_index", len(tasks)))
                prompt = record["task"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"Invalid task description for index {index}: {prompt!r}")
                tasks[index] = prompt
        return tasks

    parquet_path = meta / "tasks.parquet"
    if not parquet_path.exists():
        return tasks

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(parquet_path)
    available = set(schema.names)
    if "task_index" not in available:
        raise ValueError(f"{parquet_path} has no task_index column; found {sorted(available)}")

    text_column = next(
        (key for key in ("task", "__index_level_0__") if key in available),
        None,
    )
    if text_column is None:
        raise ValueError(
            f"{parquet_path} has no task text column (task or __index_level_0__); "
            f"found {sorted(available)}"
        )
    text_type = schema.field(text_column).type
    if not (pa.types.is_string(text_type) or pa.types.is_large_string(text_type)):
        raise ValueError(f"{parquet_path} task column {text_column} must contain strings, got {text_type}")

    columns = ["task_index", text_column]
    for record in pq.read_table(parquet_path, columns=columns).to_pylist():
        index = int(record["task_index"])
        prompt = record[text_column]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Invalid task description for index {index}: {prompt!r}")
        tasks[index] = prompt
    return tasks
