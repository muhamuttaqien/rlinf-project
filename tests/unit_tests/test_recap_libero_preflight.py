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

"""Focused CPU tests for LIBERO RECAP static data layout checks."""

import json

from toolkits.recap_libero_preflight import DATASETS, check_data_root


def _make_dataset(root, name):
    meta = root / name / "meta"
    meta.mkdir(parents=True)
    (root / name / "data").mkdir()
    (meta / "info.json").write_text(
        json.dumps({"fps": 10, "total_episodes": 1, "features": {}}),
        encoding="utf-8",
    )


def test_preflight_accepts_minimal_metadata(tmp_path):
    for name in DATASETS:
        _make_dataset(tmp_path, name)
    assert check_data_root(tmp_path) == []


def test_preflight_reports_missing_root(tmp_path):
    problems = check_data_root(tmp_path / "not-present")
    assert len(problems) == 1
    assert "root" in problems[0]


def test_preflight_reports_missing_dataset(tmp_path):
    for name in DATASETS[:2]:
        _make_dataset(tmp_path, name)
    assert "libero10_task0_eval: missing dataset directory" in check_data_root(tmp_path)


def test_preflight_reports_invalid_json(tmp_path):
    for name in DATASETS:
        _make_dataset(tmp_path, name)
    (tmp_path / DATASETS[0] / "meta" / "info.json").write_text("{", encoding="utf-8")
    assert any("invalid meta/info.json" in issue for issue in check_data_root(tmp_path))


def test_preflight_reports_missing_metadata_field(tmp_path):
    for name in DATASETS:
        _make_dataset(tmp_path, name)
    (tmp_path / DATASETS[0] / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 1, "features": {}}),
        encoding="utf-8",
    )
    assert any("missing metadata field fps" in issue for issue in check_data_root(tmp_path))
