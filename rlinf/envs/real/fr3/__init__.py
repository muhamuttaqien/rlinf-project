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

"""Offline FR3 registration; importing this package never loads physical tasks."""

from rlinf.envs.real.registry import register_tasks

from .offline import FR3OfflineRealWorldEnv, FR3OfflineTask

TASKS = {"FR3OfflineCollect-v0": FR3OfflineTask}
__all__ = [
    "FR3OfflineRealWorldEnv",
    "FR3OfflineTask",
    *register_tasks(__name__, globals(), TASKS),
]
