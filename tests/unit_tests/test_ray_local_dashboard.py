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

"""Local-only Ray dashboard startup policy."""

import pytest

from rlinf.scheduler.cluster.cluster import _local_ray_dashboard_options


@pytest.mark.parametrize("setting", ["1", "true", "yes", "on"])
def test_explicit_local_ray_may_disable_dashboard(monkeypatch, setting):
    monkeypatch.setenv("RAY_ADDRESS", "local")
    monkeypatch.setenv("RLINF_RAY_DISABLE_DASHBOARD", setting)
    assert _local_ray_dashboard_options() == {"include_dashboard": False}


@pytest.mark.parametrize("address", ["auto", "10.42.0.1:6379", ""])
def test_shared_ray_is_never_modified(monkeypatch, address):
    monkeypatch.setenv("RAY_ADDRESS", address)
    monkeypatch.setenv("RLINF_RAY_DISABLE_DASHBOARD", "1")
    assert _local_ray_dashboard_options() == {}


def test_local_ray_retains_default_without_opt_in(monkeypatch):
    monkeypatch.setenv("RAY_ADDRESS", "local")
    monkeypatch.delenv("RLINF_RAY_DISABLE_DASHBOARD", raising=False)
    assert _local_ray_dashboard_options() == {}


@pytest.mark.parametrize("count", ["1", "2"])
def test_local_ray_explicit_gpu_resources(monkeypatch, count):
    from rlinf.scheduler.cluster.cluster import _local_ray_gpu_options

    monkeypatch.setenv("RAY_ADDRESS", "local")
    monkeypatch.setenv("RLINF_RAY_LOCAL_NUM_GPUS", count)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(int(count)))))
    assert _local_ray_gpu_options() == {"num_gpus": int(count)}


def test_explicit_gpu_resources_never_apply_to_shared_cluster(monkeypatch):
    from rlinf.scheduler.cluster.cluster import _local_ray_gpu_options

    monkeypatch.setenv("RAY_ADDRESS", "auto")
    monkeypatch.setenv("RLINF_RAY_LOCAL_NUM_GPUS", "1")
    assert _local_ray_gpu_options() == {}


def test_local_ray_gpu_count_must_match_visibility(monkeypatch):
    from rlinf.scheduler.cluster.cluster import _local_ray_gpu_options

    monkeypatch.setenv("RAY_ADDRESS", "local")
    monkeypatch.setenv("RLINF_RAY_LOCAL_NUM_GPUS", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="requires exactly"):
        _local_ray_gpu_options()


def test_nvidia_probe_falls_back_to_explicit_local_count(monkeypatch):
    from ray._private.accelerators.nvidia_gpu import (
        NvidiaGPUAcceleratorManager as RayNvidia,
    )

    from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

    monkeypatch.setenv("RAY_ADDRESS", "local")
    monkeypatch.setenv("RLINF_RAY_LOCAL_NUM_GPUS", "1")
    monkeypatch.setattr(RayNvidia, "get_current_node_num_accelerators", lambda: 0)
    assert NvidiaGPUManager.get_num_devices() == 1


def test_nvidia_probe_does_not_override_shared_cluster(monkeypatch):
    from ray._private.accelerators.nvidia_gpu import (
        NvidiaGPUAcceleratorManager as RayNvidia,
    )

    from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

    monkeypatch.setenv("RAY_ADDRESS", "auto")
    monkeypatch.setenv("RLINF_RAY_LOCAL_NUM_GPUS", "1")
    monkeypatch.setattr(RayNvidia, "get_current_node_num_accelerators", lambda: 0)
    assert NvidiaGPUManager.get_num_devices() == 0
