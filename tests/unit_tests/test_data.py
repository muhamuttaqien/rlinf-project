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

"""Datasets, batching, and the on-the-wire shapes they produce."""

import asyncio
import copy
import inspect
import json
import random
import shutil
import time
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

import rlinf.data.datasets.d4rl as d4rl_dataset_module
import rlinf.utils.obs_compression as obs_compression
from rlinf.data.datasets.d4rl import D4RLDataset
from rlinf.data.datasets.reasoning.dataset import ReasoningDataset
from rlinf.data.schema.embodied_trajectory import (
    LeRobotEpisodeAccumulator,
    RolloutGeometry,
    TrajectoryCollector,
    TrajectoryMode,
    TrajectoryPlan,
    select_trajectory_collector,
    select_trajectory_dispatcher,
)
from rlinf.data.schema.embodied_types import (
    EnvOutput,
    EnvPart,
    EnvTransition,
    LeRobotFrame,
    PolicyInput,
    PolicyOutput,
    PolicyPart,
    Trajectory,
    TrajectoryKey,
    TrajectorySource,
    TrajectoryStep,
    merge_batch_values,
    merge_episode_data,
    split_batch_value,
    split_episode_data,
)
from rlinf.data.storage.lerobot import add_frame_to_dataset, episode_boundaries
from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.envs.wrappers.collect_episode import CollectEpisode
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.scheduler.channel.channel import DEFAULT_KEY
from rlinf.scheduler.channel.hooks import ChannelContext
from rlinf.scheduler.cluster.utils import (
    TensorPlaceholder,
    pack_dataclass_tensors,
    unpack_dataclass_tensors,
)
from rlinf.utils.env_helpers import SmoothInterveneController
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.utils.obs_compression import (
    _CODEC_KEY,
    compress_obs,
    decompress_obs,
    infer_obs_batch_size,
    is_compressed_image,
    is_compression_enabled,
)
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _scalar_trajectory(value: float) -> Trajectory:
    """One-sample trajectory so checkpoint tests write only tiny files."""
    return Trajectory(max_episode_length=1, rewards=torch.full((1, 1, 1), float(value)))


def test_memory_checkpoint_indexes_only_cached_trajectories(tmp_path):
    """A memory checkpoint records only the trajectories still held in cache."""
    checkpoint = tmp_path / "checkpoint"
    buffer = TrajectoryReplayBuffer(sample_window_size=2)
    restored = None
    try:
        for value in range(3):
            buffer.add_trajectories([_scalar_trajectory(value)])
        original_stats = buffer.get_stats()
        buffer.save_checkpoint(str(checkpoint))
        assert buffer.get_stats() == original_stats

        index = json.loads((checkpoint / "trajectory_index.json").read_text())
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        assert index["trajectory_id_list"] == [1, 2]
        assert metadata["size"] == 2
        assert metadata["total_samples"] == 2
        assert metadata["trajectory_counter"] == 3
        assert len(list(checkpoint.glob("trajectory_*.pt"))) == 2

        restored = TrajectoryReplayBuffer(sample_window_size=2)
        restored.load_checkpoint(str(checkpoint))
        assert restored.get_stats()["num_trajectories"] == 2
        assert restored.get_stats()["total_samples"] == 2
    finally:
        buffer.close()
        if restored is not None:
            restored.close()
        shutil.rmtree(checkpoint, ignore_errors=True)


def test_auto_save_checkpoint_copies_current_window(tmp_path):
    """An auto-save checkpoint copies the current window, not a stale sample window."""
    source = tmp_path / "buffer"
    checkpoint = tmp_path / "checkpoint"
    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(source),
        cache_size=3,
        sample_window_size=2,
    )
    restored = None
    try:
        buffer.add_trajectories([_scalar_trajectory(0)])
        buffer.add_trajectories([_scalar_trajectory(1)])
        buffer.sample(1)
        buffer.add_trajectories([_scalar_trajectory(2)])
        original_stats = buffer.get_stats()
        buffer.save_checkpoint(str(checkpoint))
        assert buffer.get_stats() == original_stats

        index = json.loads((checkpoint / "trajectory_index.json").read_text())
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        assert index["trajectory_id_list"] == [1, 2]
        assert metadata["size"] == 2
        assert metadata["total_samples"] == 2
        assert metadata["trajectory_counter"] == 3
        assert len(list(checkpoint.glob("trajectory_*.pt"))) == 2

        restored = TrajectoryReplayBuffer(sample_window_size=2)
        restored.load_checkpoint(str(checkpoint))
        assert restored.get_stats()["num_trajectories"] == 2
        assert restored.get_stats()["total_samples"] == 2
    finally:
        buffer.close()
        if restored is not None:
            restored.close()
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(checkpoint, ignore_errors=True)


def test_auto_save_checkpoint_resaves_restored_window(tmp_path):
    """A resumed auto-save buffer copies old and newly written trajectories."""
    source = tmp_path / "source"
    first_checkpoint = tmp_path / "first_checkpoint"
    resumed_source = tmp_path / "resumed_source"
    second_checkpoint = tmp_path / "second_checkpoint"
    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(source),
        sample_window_size=2,
    )
    resumed = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(resumed_source),
        sample_window_size=2,
    )
    restored = None
    try:
        for value in range(3):
            buffer.add_trajectories([_scalar_trajectory(value)])
        buffer.save_checkpoint(str(first_checkpoint))

        resumed.load_checkpoint(str(first_checkpoint))
        resumed.add_trajectories([_scalar_trajectory(3)])
        resumed.save_checkpoint(str(second_checkpoint))

        index = json.loads((second_checkpoint / "trajectory_index.json").read_text())
        metadata = json.loads((second_checkpoint / "metadata.json").read_text())
        assert index["trajectory_id_list"] == [2, 3]
        assert metadata["size"] == 2
        assert metadata["total_samples"] == 2
        assert metadata["trajectory_counter"] == 4
        assert len(list(second_checkpoint.glob("trajectory_*.pt"))) == 2

        restored = TrajectoryReplayBuffer(sample_window_size=2)
        restored.load_checkpoint(str(second_checkpoint))
        assert restored.get_stats()["num_trajectories"] == 2
        assert restored.get_stats()["total_samples"] == 2
    finally:
        buffer.close()
        resumed.close()
        if restored is not None:
            restored.close()
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(first_checkpoint, ignore_errors=True)
        shutil.rmtree(resumed_source, ignore_errors=True)
        shutil.rmtree(second_checkpoint, ignore_errors=True)


class TestD4RLDataset:
    """Tests for loading D4RL transition datasets."""

    @pytest.mark.parametrize("has_next_observations", [False, True])
    def test_from_path_converts_standard_hdf5_dataset(
        self, tmp_path, monkeypatch, has_next_observations
    ):
        """Standard D4RL files go through ``qlearning_dataset``.

        AntMaze files have no ``next_observations``; MuJoCo v2 files have both
        ``next_observations`` and ``timeouts``.
        """
        dataset_path = tmp_path / "standard-d4rl.hdf5"
        dataset_path.touch()
        raw = {
            "observations": np.array([[0.0], [1.0], [10.0], [11.0], [12.0]]),
            "actions": np.array([[-1.5], [0.25], [0.5], [0.75], [1.5]]),
            "rewards": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "terminals": np.array([False, False, False, False, True]),
            "timeouts": np.array([False, True, False, False, False]),
        }
        if has_next_observations:
            raw["next_observations"] = np.array([[1.0], [2.0], [11.0], [12.0], [13.0]])
        converted = {
            "observations": raw["observations"][[0, 2, 3]],
            "actions": raw["actions"][[0, 2, 3]],
            "rewards": raw["rewards"][[0, 2, 3]],
            "terminals": raw["terminals"][[0, 2, 3]],
            "next_observations": raw["observations"][[1, 3, 4]],
        }

        env = mock.Mock()
        env.get_dataset.return_value = raw
        gym_api = mock.Mock()
        gym_api.make.return_value = env

        def qlearning_dataset(actual_env, *, dataset):
            assert actual_env is env
            assert dataset is raw
            assert "timeouts" in dataset
            return converted

        d4rl_api = mock.Mock(qlearning_dataset=qlearning_dataset)
        monkeypatch.setattr(d4rl_dataset_module, "gym", gym_api)
        monkeypatch.setattr(d4rl_dataset_module, "d4rl", d4rl_api)

        dataset = D4RLDataset.from_path(dataset_path, task_name="antmaze-test-v0")

        np.testing.assert_allclose(dataset.observations[:, 0], [0.0, 10.0, 11.0])
        np.testing.assert_allclose(dataset.next_observations[:, 0], [1.0, 11.0, 12.0])
        np.testing.assert_allclose(dataset.actions[:, 0], [-0.99999, 0.5, 0.75])
        np.testing.assert_allclose(dataset.rewards, [0.0, 2.0, 3.0])
        np.testing.assert_allclose(dataset.dones_float, [1.0, 0.0, 1.0])
        env.get_dataset.assert_called_once_with(h5path=str(dataset_path))
        env.close.assert_called_once_with()

    def test_from_path_preserves_materialized_transition_dataset(
        self, tmp_path, monkeypatch
    ):
        """Existing files with next observations remain supported."""
        dataset_path = tmp_path / "materialized-d4rl.hdf5"
        dataset_path.touch()
        raw = {
            "observations": np.array([[1.0], [2.0]]),
            "actions": np.array([[-0.25], [0.25]]),
            "rewards": np.array([3.0, 4.0]),
            "terminals": np.array([False, True]),
            "next_observations": np.array([[2.0], [3.0]]),
        }

        env = mock.Mock()
        env.get_dataset.return_value = raw
        gym_api = mock.Mock()
        gym_api.make.return_value = env
        d4rl_api = mock.Mock()
        monkeypatch.setattr(d4rl_dataset_module, "gym", gym_api)
        monkeypatch.setattr(d4rl_dataset_module, "d4rl", d4rl_api)

        dataset = D4RLDataset.from_path(dataset_path, task_name="custom-test-v0")

        np.testing.assert_array_equal(dataset.observations, raw["observations"])
        np.testing.assert_array_equal(
            dataset.next_observations, raw["next_observations"]
        )
        d4rl_api.qlearning_dataset.assert_not_called()
        env.close.assert_called_once_with()


class TestMathDatasetMultithread:
    """Tests for ReasoningDataset multithread processing consistency."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = mock.Mock()
        tokenizer.is_fast = True
        tokenizer.eos_token_id = 2

        def apply_chat_template_side_effect(
            prompts, tokenize=False, add_generation_prompt=True
        ):
            """Mock apply_chat_template that handles generator input."""
            # Convert generator to list if needed
            prompts_list = list(prompts) if not isinstance(prompts, list) else prompts
            return [
                f"<|user|>\n{prompt}\n<|assistant|>\n"
                if isinstance(prompt, str)
                else prompt
                for prompt in prompts_list
            ]

        tokenizer.apply_chat_template = mock.Mock(
            side_effect=apply_chat_template_side_effect
        )
        tokenizer.batch_encode_plus = mock.Mock(
            side_effect=lambda texts: {
                "input_ids": [[1] * len(text.split()) for text in texts]
            }
        )
        tokenizer.encode = mock.Mock(side_effect=lambda text: [1] * len(text.split()))
        return tokenizer

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = DictConfig(
            {
                "data": {
                    "max_prompt_length": 1000,
                    "prompt_key": "question",
                    "answer_key": "answer",
                    "apply_chat_template": True,
                    "filter_prompt_by_length": False,
                    "process_workers": 4,
                    "process_batch_size": 32,
                }
            }
        )
        return config

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing (at least 10000 items)."""
        # Generate at least 10000 math problems
        data = []
        operations = [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
            ("/", lambda a, b: a // b if b != 0 else 0),
        ]

        for i in range(10000):
            op_symbol, op_func = random.choice(operations)
            a = random.randint(1, 1000)
            b = random.randint(1, 1000) if op_symbol != "/" else random.randint(1, 100)
            if op_symbol == "/" and b == 0:
                b = 1
            result = op_func(a, b)
            question = f"What is {a} {op_symbol} {b}?"
            data.append({"question": question, "answer": str(result)})

        return data

    def test_multithread_vs_singlethread_consistency(
        self, mock_tokenizer, mock_config, sample_data, tmp_path
    ):
        """
        Test that multithread processing produces identical results to single-thread processing.

        This test verifies that:
        1. Results from multi-worker processing match single-worker processing
        2. All keys are the same
        3. All values are the same
        """
        # Create a temporary JSON file with sample data
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Create ReasoningDataset instance to get the configuration
        dataset = ReasoningDataset(
            data_paths=str(data_file),
            config=mock_config,
            tokenizer=mock_tokenizer,
        )

        # Use original raw data (before processing) for testing
        # We need to reload the raw data to avoid double processing
        raw_data = dataset._load_data()

        # Deep copy to avoid modifying the original
        raw_data_multithread = copy.deepcopy(raw_data)
        raw_data_singlethread = copy.deepcopy(raw_data)

        # Test with multithread parameters
        time_start = time.time()
        data_multithread = dataset.load_post_process(
            raw_data_multithread, dataset.process_workers, dataset.process_batch_size
        )
        time_elapse_multithread = time.time() - time_start

        # Test with single thread
        time_start = time.time()
        data_singlethread = dataset.load_post_process(raw_data_singlethread, 1, 1)
        time_elapse_singlethread = time.time() - time_start

        # Verify lengths are equal
        assert len(data_multithread) == len(data_singlethread), (
            f"Length mismatch: multithread={len(data_multithread)}, singlethread={len(data_singlethread)}"
        )

        # Verify all items have the same keys and values
        for idx, (item_mt, item_st) in enumerate(
            zip(data_multithread, data_singlethread)
        ):
            keys_mt, keys_st = item_mt.keys(), item_st.keys()
            assert keys_mt == keys_st, (
                f"Keys mismatch at index {idx}: "
                f"multithread={list(keys_mt)}, singlethread={list(keys_st)}"
            )

            # Check all values are equal
            unequal_keys = [key for key in keys_mt if item_mt[key] != item_st[key]]
            assert len(unequal_keys) == 0, (
                f"Values mismatch at index {idx} for keys: {unequal_keys}"
            )

        # Log timing information (for debugging)
        print(
            f"Data count: {len(data_multithread)}, "
            f"Multithread processing time: {time_elapse_multithread:.2f}s, "
            f"Singlethread processing time: {time_elapse_singlethread:.2f}s"
        )

    def test_multithread_consistency_with_filter(
        self, mock_tokenizer, mock_config, sample_data, tmp_path
    ):
        """
        Test multithread processing consistency when filter_prompt_by_length is enabled.
        """
        # Update config to enable filtering
        mock_config.data.filter_prompt_by_length = True
        mock_config.data.max_prompt_length = 50  # Reasonable limit to test filtering

        # Create a temporary JSON file with sample data
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Create ReasoningDataset instance to get the configuration
        dataset = ReasoningDataset(
            data_paths=str(data_file),
            config=mock_config,
            tokenizer=mock_tokenizer,
        )

        # Use original raw data (before processing) for testing
        raw_data = dataset._load_data()

        # Deep copy to avoid modifying the original
        raw_data_multithread = copy.deepcopy(raw_data)
        raw_data_singlethread = copy.deepcopy(raw_data)

        # Test with multithread parameters
        data_multithread = dataset.load_post_process(
            raw_data_multithread, dataset.process_workers, dataset.process_batch_size
        )

        # Test with single thread
        data_singlethread = dataset.load_post_process(raw_data_singlethread, 1, 1)

        # Verify consistency
        assert len(data_multithread) == len(data_singlethread), (
            f"Length mismatch: multithread={len(data_multithread)}, singlethread={len(data_singlethread)}"
        )

        # Verify that some data was filtered (not all data passed)
        assert len(data_multithread) <= len(raw_data), (
            f"Filtering should reduce data size, but got {len(data_multithread)} >= {len(raw_data)}"
        )

        for idx, (item_mt, item_st) in enumerate(
            zip(data_multithread, data_singlethread)
        ):
            assert item_mt.keys() == item_st.keys(), f"Keys mismatch at index {idx}"
            for key in item_mt.keys():
                assert item_mt[key] == item_st[key], (
                    f"Mismatch at index {idx}, key {key}"
                )

        print(
            f"Filtered data count: {len(data_multithread)}/{len(raw_data)} "
            f"(max_prompt_length={mock_config.data.max_prompt_length})"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])


def test_split_dict_to_chunk_keeps_mixed_fields_aligned():
    batch = {
        "values": torch.arange(10),
        "sample_ids": list(range(10)),
        "nested": {"values": torch.arange(10) + 100},
    }

    chunks = split_dict_to_chunk(batch, 3)

    assert [chunk["values"].tolist() for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert [chunk["sample_ids"] for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert [chunk["nested"]["values"].tolist() for chunk in chunks] == [
        [100, 101, 102, 103],
        [104, 105, 106],
        [107, 108, 109],
    ]


def test_split_dict_to_chunk_returns_requested_number_of_chunks():
    batch = {"values": torch.arange(2), "sample_ids": ["a", "b"]}

    chunks = split_dict_to_chunk(batch, 4)

    assert len(chunks) == 4
    assert [chunk["values"].tolist() for chunk in chunks] == [[0], [1], [], []]
    assert [chunk["sample_ids"] for chunk in chunks] == [["a"], ["b"], [], []]


class _LegacyDataset:
    """Mimics lerobot < 0.2: the task lives inside the frame dict."""

    def __init__(self):
        self.frames = []
        self.saved_episodes = 0

    def add_frame(self, frame):
        if "task" not in frame:
            raise ValueError("Missing features: {'task'}")
        self.frames.append(frame)

    def save_episode(self):
        self.saved_episodes += 1


class _PostRevertDataset(_LegacyDataset):
    """Mimics lerobot >= 0.4: back to ``add_frame(frame)``, but it pops the task.

    0.4 reverted the 0.3.x signature, so a version-number check would dispatch
    this one wrongly. It also mutates the caller's dict.
    """

    def add_frame(self, frame):
        if "task" not in frame:
            raise ValueError("Missing features: {'task'}")
        self.frames.append({**frame, "task": frame.pop("task")})


class _CurrentDataset:
    """Mimics lerobot >= 0.3: the task is a separate argument.

    Like the real implementation, a ``task`` key inside *frame* is rejected
    because it is not part of the feature schema.
    """

    def __init__(self):
        self.frames = []
        self.tasks = []
        self.saved_episodes = 0

    def add_frame(self, frame, task, timestamp=None):
        if "task" in frame:
            raise ValueError("Extra features: {'task'}")
        self.frames.append(frame)
        self.tasks.append(task)

    def save_episode(self):
        self.saved_episodes += 1


def _make_writer(dataset):
    # ``create()`` needs a real lerobot install, so attach the dataset the way
    # ``create()`` would.
    writer = LeRobotDatasetWriter()
    writer.dataset = dataset
    return writer


def _episode(n=2):
    return [{"state": i, "actions": i, "task": "pick up the cube"} for i in range(n)]


def test_legacy_dataset_keeps_task_in_frame():
    dataset = _LegacyDataset()
    _make_writer(dataset).add_episode(_episode())

    assert [f["task"] for f in dataset.frames] == ["pick up the cube"] * 2
    assert dataset.saved_episodes == 1


def test_current_dataset_gets_task_as_argument():
    dataset = _CurrentDataset()
    _make_writer(dataset).add_episode(_episode())

    assert dataset.tasks == ["pick up the cube"] * 2
    assert all("task" not in f for f in dataset.frames)
    assert dataset.frames[0]["state"] == 0
    assert dataset.saved_episodes == 1


ALL_SHAPES = [_LegacyDataset, _CurrentDataset, _PostRevertDataset]


def test_post_revert_dataset_keeps_task_in_frame():
    # lerobot >= 0.4 took the 0.3.x signature back out again.
    dataset = _PostRevertDataset()
    _make_writer(dataset).add_episode(_episode())

    assert [f["task"] for f in dataset.frames] == ["pick up the cube"] * 2
    assert dataset.saved_episodes == 1


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_caller_frames_are_not_mutated(dataset_cls):
    # The DAgger worker shares these dicts with the in-memory training store,
    # and lerobot >= 0.4 pops "task" out of whatever frame it is handed.
    episode = _episode()
    before = [dict(f) for f in episode]
    _make_writer(dataset_cls()).add_episode(episode)

    assert episode == before


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_frame_without_task_is_rejected(dataset_cls):
    with pytest.raises(ValueError, match="missing the required 'task' field"):
        add_frame_to_dataset(dataset_cls(), {"state": 0, "actions": 0})


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_add_frame_to_dataset_is_usable_standalone(dataset_cls):
    # The toolkit collectors drive LeRobotDataset directly, without the writer.
    dataset = dataset_cls()
    add_frame_to_dataset(dataset, {"state": 0, "task": "wipe the table"})

    assert len(dataset.frames) == 1


def test_empty_episode_is_skipped():
    dataset = _CurrentDataset()
    _make_writer(dataset).add_episode([])

    assert dataset.frames == []
    assert dataset.saved_episodes == 0


# --------------------------------------------------------------------------
# episode_boundaries: dataset format v2.1 vs v3.0
# --------------------------------------------------------------------------


class _V21Dataset:
    """Dataset format v2.1: a dict of two tensors on the dataset itself."""

    def __init__(self, starts, ends):
        self.episode_data_index = {
            "from": torch.tensor(starts),
            "to": torch.tensor(ends),
        }


class _V30Meta:
    def __init__(self, starts, ends):
        self.episodes = {"dataset_from_index": starts, "dataset_to_index": ends}


class _V30Dataset:
    """Dataset format v3.0 (lerobot >= 0.4): columns on ``meta.episodes``."""

    def __init__(self, starts, ends):
        self.episode_data_index = None
        self.meta = _V30Meta(starts, ends)


@pytest.mark.parametrize("dataset_cls", [_V21Dataset, _V30Dataset])
def test_episode_boundaries_agree_across_formats(dataset_cls):
    starts, ends = episode_boundaries(dataset_cls([0, 3, 7], [3, 7, 9]))

    assert starts == [0, 3, 7]
    assert ends == [3, 7, 9]
    assert all(isinstance(x, int) for x in starts + ends)


def test_episode_boundaries_reports_an_unknown_layout():
    class _Alien:
        episode_data_index = None
        meta = None

    with pytest.raises(RuntimeError, match="Cannot determine episode boundaries"):
        episode_boundaries(_Alien())


def test_episode_boundaries_rejects_v30_meta_without_the_columns():
    class _Partial:
        episode_data_index = None
        meta = _V30Meta([0], [1])

    _Partial.meta.episodes = {"length": [1]}

    with pytest.raises(RuntimeError, match="Cannot determine episode boundaries"):
        episode_boundaries(_Partial())


# Skip codec round-trip tests when the optional backends are not installed.
_CODECS = []
try:
    import lz4.frame  # noqa: F401

    _CODECS.append("lz4")
except ImportError:
    pass
try:
    import zstandard  # noqa: F401

    _CODECS.append("zstd")
except ImportError:
    pass

requires_codec = pytest.mark.skipif(
    not _CODECS, reason="no observation compression codec (lz4/zstd) installed"
)


def _make_payload(num_envs: int = 4) -> dict:
    """A payload shaped like EnvWorker._build_rollout_input_data output."""
    obs = {
        "main_images": torch.randint(0, 256, (num_envs, 8, 8, 3), dtype=torch.uint8),
        "extra_view_images": torch.randint(
            0, 256, (num_envs, 6, 6, 3), dtype=torch.uint8
        ),
        "states": torch.randn(num_envs, 7, dtype=torch.float32),
        "task_descriptions": ["put carrot on plate"] * num_envs,
    }
    return {
        "obs": obs,
        "final_obs": {
            "main_images": torch.randint(
                0, 256, (num_envs, 8, 8, 3), dtype=torch.uint8
            ),
            "states": torch.randn(num_envs, 7, dtype=torch.float32),
        },
        "rlt_switch_flags": None,
    }


def _assert_payload_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, dict):
            _assert_payload_equal(va, vb)
        elif isinstance(va, torch.Tensor):
            assert torch.equal(va, vb), f"tensor mismatch for {key!r}"
        else:
            assert va == vb, f"value mismatch for {key!r}"


def _cfg(**overrides):
    # A plain dict is sufficient: the codec only calls ``config.get(...)``,
    # which both ``dict`` and OmegaConf's ``DictConfig`` support identically.
    base = {"enable": True, "codec": "lz4", "level": 1, "xor_delta": True}
    base.update(overrides)
    return base


@requires_codec
@pytest.mark.parametrize("codec", _CODECS)
@pytest.mark.parametrize("xor_delta", [True, False])
def test_compress_decompress_is_lossless(codec, xor_delta):
    payload = _make_payload()
    config = _cfg(codec=codec, xor_delta=xor_delta)

    compressed = compress_obs(payload, config)
    # Image tensors are replaced by self-describing marker dicts...
    assert _CODEC_KEY in compressed["obs"]["main_images"]
    assert _CODEC_KEY in compressed["obs"]["extra_view_images"]
    # ...while non-image fields are passed through untouched.
    assert isinstance(compressed["obs"]["states"], torch.Tensor)
    assert compressed["obs"]["task_descriptions"] == payload["obs"]["task_descriptions"]

    restored = decompress_obs(compressed)
    _assert_payload_equal(payload, restored)


@requires_codec
def test_single_env_batch_roundtrip():
    # XOR-delta is skipped when there is only one frame; must still be lossless.
    payload = _make_payload(num_envs=1)
    restored = decompress_obs(compress_obs(payload, _cfg(xor_delta=True)))
    _assert_payload_equal(payload, restored)


def test_disabled_config_is_passthrough():
    payload = _make_payload()
    assert compress_obs(payload, _cfg(enable=False)) is payload
    assert compress_obs(payload, None) is payload
    assert not is_compression_enabled(None)
    assert not is_compression_enabled(_cfg(enable=False))
    assert is_compression_enabled(_cfg(enable=True))


def test_decompress_on_uncompressed_payload_is_noop():
    # The rollout worker always routes received data through decompress_obs, so
    # it must be a no-op on payloads sent without compression.
    payload = _make_payload()
    restored = decompress_obs(payload)
    _assert_payload_equal(payload, restored)


@requires_codec
def test_only_uint8_images_are_compressed():
    # A float image-shaped tensor is not a uint8 observation and must be left
    # untouched, as must low-rank uint8 tensors (e.g. flags).
    payload = {
        "obs": {
            "float_map": torch.randn(4, 8, 8, 3),
            "uint8_flags": torch.ones(4, dtype=torch.uint8),
        }
    }
    compressed = compress_obs(payload, _cfg())
    assert isinstance(compressed["obs"]["float_map"], torch.Tensor)
    assert isinstance(compressed["obs"]["uint8_flags"], torch.Tensor)


def test_unknown_codec_raises():
    payload = _make_payload()
    with pytest.raises(ValueError, match="Unknown observation compression codec"):
        compress_obs(payload, _cfg(codec="bogus"))


@requires_codec
@pytest.mark.parametrize("codec", _CODECS)
def test_routing_split_then_compress_roundtrip(codec):
    """Compression must be compatible with the Env->Rollout channel routing.

    The env worker installs compression as a ``split_fn`` so it runs *after*
    the scheduler splits the batch: ``infer_batch_size`` and ``split_batch``
    see plain tensors, and each shard is compressed independently. This test
    reproduces that flow with the real routing helpers and asserts the payload
    survives split -> compress -> decompress -> merge unchanged.
    """
    routing = pytest.importorskip("rlinf.scheduler.worker.routing")

    payload = _make_payload(num_envs=6)
    # The scheduler infers the batch size from the *uncompressed* payload.
    assert routing.infer_batch_size(payload) == 6

    # split_fn = split_batch first, then compress each shard (env send path).
    split_sizes = [2, 1, 3]
    shards = routing.split_batch(payload, split_sizes)
    compressed_shards = [compress_obs(shard, _cfg(codec=codec)) for shard in shards]

    # Rollout side: decompress each shard, then merge (merge_obs path).
    restored_shards = [decompress_obs(shard) for shard in compressed_shards]
    merged = routing.merge_batches(restored_shards)
    _assert_payload_equal(payload, merged)


def test_infer_obs_batch_size_uncompressed():
    payload = _make_payload(num_envs=5)
    assert infer_obs_batch_size(payload) == 5
    # Also accepts a bare obs dict (no "obs" wrapper).
    assert infer_obs_batch_size(payload["obs"]) == 5


@requires_codec
def test_infer_obs_batch_size_with_compressed_images():
    # The rollout worker infers batch size on the receive path, before
    # decompression, so a compressed image must still report its batch size.
    payload = _make_payload(num_envs=5)
    compressed = compress_obs(payload, _cfg())
    assert is_compressed_image(compressed["obs"]["main_images"])
    assert infer_obs_batch_size(compressed) == 5


@requires_codec
def test_infer_obs_batch_size_images_only():
    # Regression: a batch whose only batched field is a (compressed) image,
    # with no states/task_descriptions, must not break batch-size inference.
    payload = {
        "obs": {
            "main_images": torch.randint(0, 256, (3, 8, 8, 3), dtype=torch.uint8),
        }
    }
    compressed = compress_obs(payload, _cfg())
    assert infer_obs_batch_size(compressed) == 3


def test_infer_obs_batch_size_raises_when_unbatched():
    with pytest.raises(ValueError, match="Cannot infer batch size"):
        infer_obs_batch_size({"obs": {}})


def test_env_output_composes_one_transition_object():
    transition = EnvTransition(
        rewards=torch.ones(2, 1),
        dones=torch.zeros(2, 1, dtype=torch.bool),
    )
    output = EnvOutput(obs={"states": torch.zeros(2, 3)}, transition=transition)

    assert output.transition is transition
    assert output.rewards is transition.rewards
    assert output.dones is transition.dones
    assert set(output.__dataclass_fields__) == {
        "obs",
        "transition",
        "final_obs",
        "env_infos",
    }


def test_removed_duplicate_types_are_not_schema_api():
    import rlinf.data.schema as schema

    for name in (
        "ChunkStepResult",
        "DummyPolicyInput",
        "EmbodiedRolloutResult",
        "EnvResult",
        "PolicyCompletion",
    ):
        assert not hasattr(schema, name)


def test_env_part_completion_reuses_the_environment_payload():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    transition = EnvTransition(rewards=torch.ones(2, 1))
    part = EnvPart(
        sources=[TrajectorySource(key, 2)],
        transition=transition,
        next_obs={"states": torch.zeros(2, 3)},
        requires_inference=True,
    )

    completed = part.complete(
        next_obs=part.next_obs,
        next_rlt_obs={"states": torch.ones(2, 3)},
        final_prev_values=torch.tensor([[2.0], [3.0]]),
    )

    assert completed.transition is transition
    assert not completed.requires_inference
    assert torch.equal(completed.bootstrap_values, torch.tensor([[2.0], [3.0]]))
    assert torch.equal(completed.final_prev_values, completed.bootstrap_values)


def test_transport_results_store_contiguous_cpu_tensors():
    non_contiguous = torch.arange(12).reshape(3, 4).T

    env_transition = EnvTransition(rewards=non_contiguous)
    policy_output = PolicyOutput(
        forward_inputs={"states": non_contiguous},
    )

    assert env_transition.rewards.device.type == "cpu"
    assert env_transition.rewards.is_contiguous()
    assert not hasattr(policy_output, "actions")
    assert policy_output.forward_inputs["states"].device.type == "cpu"
    assert policy_output.forward_inputs["states"].is_contiguous()


def test_policy_output_detaches_nested_model_tensors_for_transport():
    source = torch.arange(12.0, requires_grad=True)
    model_output = (source * 2).reshape(3, 4).T
    policy_output = PolicyOutput(
        forward_inputs={
            "nested": {
                "states": model_output,
                "features": [model_output[:, :2]],
            }
        },
        prev_logprobs=model_output,
        prev_values=model_output[:, :1],
        versions=model_output,
    )
    part = PolicyPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 4)],
        obs={},
        output=policy_output,
    )

    _, tensors = pack_dataclass_tensors(part)
    assert tensors
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert all(tensor.is_contiguous() for tensor in tensors)
    assert all(not tensor.requires_grad for tensor in tensors)
    assert all(tensor.grad_fn is None for tensor in tensors)


def test_nested_dataclass_transport_separates_tensors_from_skeleton():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    shared = torch.arange(4).reshape(2, 2)
    event = PolicyPart(
        sources=[TrajectorySource(key, 2)],
        obs={"states": shared, "task_descriptions": ["a", "b"]},
        output=PolicyOutput(
            forward_inputs={"nested": {"states": shared}},
            prev_values=torch.ones(2, 1),
        ),
    )

    skeleton, tensors = pack_dataclass_tensors(event)
    restored = unpack_dataclass_tensors(skeleton, tensors)

    assert isinstance(skeleton.obs["states"], TensorPlaceholder)
    assert isinstance(
        skeleton.output.forward_inputs["nested"]["states"],
        TensorPlaceholder,
    )
    assert len(tensors) == 2
    assert restored.obs["states"] is restored.output.forward_inputs["nested"]["states"]
    assert torch.equal(restored.output.prev_values, torch.ones(2, 1))


def test_policy_input_split_merge_preserves_sources_and_nested_payloads():
    keys = [TrajectoryKey(0, 0, rank, 0, 2) for rank in range(2)]
    policy_input = PolicyInput(
        obs={
            "states": torch.arange(12).reshape(4, 3),
            "labels": np.arange(4),
        },
        rlt_switch_flags=torch.tensor([False, True, False, True]),
        sources=[TrajectorySource(key, 2) for key in keys],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert shards[0].sources == [TrajectorySource(keys[0], 1)]
    assert shards[1].sources == [
        TrajectorySource(keys[0], 1, offset=1),
        TrajectorySource(keys[1], 2),
    ]
    assert merged.sources == policy_input.sources
    assert torch.equal(merged.obs["states"], policy_input.obs["states"])
    assert np.array_equal(merged.obs["labels"], policy_input.obs["labels"])
    assert torch.equal(merged.rlt_switch_flags, policy_input.rlt_switch_flags)


def test_external_policy_input_split_merge_preserves_actions():
    key = TrajectoryKey(0, 0, 0, 0, 1)
    policy_input = PolicyInput(
        obs={"states": torch.arange(12).reshape(4, 3)},
        external_actions=torch.arange(24).reshape(4, 2, 3),
        sources=[TrajectorySource(key, 4)],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert all(not shard.requires_inference for shard in shards)
    assert not merged.requires_inference
    assert torch.equal(merged.external_actions, policy_input.external_actions)
    assert merged.sources == policy_input.sources


def test_online_lerobot_payload_survives_source_routing():
    episode_data = {
        "chunk_actions": torch.arange(12).reshape(3, 2, 2),
        "obs_list": [
            {"images": torch.arange(6).reshape(3, 2)},
            {"images": torch.arange(6, 12).reshape(3, 2)},
        ],
        "terminations": torch.tensor([False, True, False]),
        "truncations": torch.tensor([False, False, True]),
        "infos_list": [
            {"score": np.arange(3)},
            {"score": np.arange(3, 6)},
        ],
    }
    split_data = split_episode_data(episode_data, [1, 2])
    merged_data = merge_episode_data(split_data)

    assert merged_data is not None
    assert torch.equal(merged_data["chunk_actions"], episode_data["chunk_actions"])
    assert torch.equal(
        merged_data["obs_list"][1]["images"],
        episode_data["obs_list"][1]["images"],
    )
    assert np.array_equal(
        merged_data["infos_list"][0]["score"],
        episode_data["infos_list"][0]["score"],
    )


def test_env_part_split_merge_preserves_offsets():
    current_key = TrajectoryKey(1, 2, 3, 0, 4)
    previous_key = TrajectoryKey(1, 2, 3, 0, 3)
    policy_input = PolicyInput(
        obs={"states": torch.arange(12).reshape(4, 3)},
        sources=[TrajectorySource(current_key, 4)],
        env_parts=[
            EnvPart(
                sources=[TrajectorySource(previous_key, 4)],
                transition=EnvTransition(rewards=torch.arange(4).reshape(4, 1)),
                next_obs={"states": torch.arange(12).reshape(4, 3)},
                requires_inference=False,
                initial_transition=EnvTransition(
                    dones=torch.zeros(4, 1, dtype=torch.bool)
                ),
            )
        ],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert shards[1].sources == [TrajectorySource(current_key, 3, offset=1)]
    assert shards[1].env_parts[0].sources == [
        TrajectorySource(previous_key, 3, offset=1)
    ]
    assert merged.request_sizes == [1, 3]
    assert torch.equal(
        shards[1].env_parts[0].initial_transition.dones,
        torch.zeros(3, 1, dtype=torch.bool),
    )
    assert torch.equal(
        merged.env_parts[1].next_obs["states"],
        torch.arange(12).reshape(4, 3)[1:],
    )


def test_scalar_batch_leaves_survive_a_split_merge_round_trip():
    """Scalars broadcast by a split must collapse back to the original value.

    Source fragments carry scalar info flags (e.g. ``record_reset``) unchanged,
    so reassembling them must not turn one flag into a list of per-shard copies.
    """
    for value in (True, 3, 1.5, "reset"):
        shards = split_batch_value(value, [2, 2])
        assert shards == [value, value]
        assert merge_batch_values(shards) == value

    nested = {"flag": True, "name": "abc", "states": torch.arange(4).reshape(4, 1)}
    shards = split_batch_value(nested, [3, 1])
    merged = merge_batch_values(shards)
    assert merged["flag"] is True
    assert merged["name"] == "abc"
    assert torch.equal(merged["states"], nested["states"])


def test_merging_conflicting_scalar_batch_values_is_rejected():
    with pytest.raises(ValueError, match="conflicting scalar"):
        merge_batch_values([True, False])


def test_episode_data_round_trip_keeps_scalar_info_flags_intact():
    episode_data = {
        "chunk_actions": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "obs_list": [{"states": torch.arange(4).reshape(4, 1)}],
        "terminations": torch.zeros(4, 1, dtype=torch.bool),
        "truncations": torch.zeros(4, 1, dtype=torch.bool),
        "infos_list": [{"record_reset": True, "segment_advance": False}],
    }

    merged = merge_episode_data(split_episode_data(episode_data, [3, 1]))

    assert merged["infos_list"][0]["record_reset"] is True
    assert merged["infos_list"][0]["segment_advance"] is False
    assert torch.equal(merged["chunk_actions"], episode_data["chunk_actions"])


def test_policy_part_owns_routed_fragment_split_and_merge():
    key = TrajectoryKey(1, 0, 0, 0, 2)
    actions = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    part = PolicyPart(
        sources=[TrajectorySource(key, 1), TrajectorySource(key, 3, offset=1)],
        obs={"states": torch.arange(12).reshape(4, 3)},
        output=PolicyOutput(
            forward_inputs={"action": actions},
            prev_values=torch.arange(4).reshape(4, 1),
        ),
    )

    fragments = part.split()
    merged = PolicyPart.merge(fragments)

    assert len(fragments) == 2
    assert fragments[1].sources == [TrajectorySource(key, 3, offset=1)]
    assert torch.equal(fragments[0].output.forward_inputs["action"], actions[:1])
    assert torch.equal(merged.obs["states"], part.obs["states"])


def test_trajectory_step_owns_intervention_and_transition_conversion():
    key = TrajectoryKey(1, 0, 0, 0, 2)
    model_actions = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    policy = PolicyPart(
        sources=[TrajectorySource(key, 1)],
        obs={"states": torch.zeros(1, 2), "task_descriptions": ["pick"]},
        output=PolicyOutput(
            forward_inputs={"action": model_actions, "model_action": model_actions},
            prev_values=torch.ones(1, 1),
        ),
    )
    env = EnvPart(
        sources=[TrajectorySource(key, 1)],
        transition=EnvTransition(
            rewards=torch.ones(1, 1),
            intervene_actions=torch.tensor([[9.0, 8.0, 7.0, 6.0]]),
            intervene_flags=torch.tensor([[False, True]]),
        ),
        next_obs={"states": torch.ones(1, 2), "task_descriptions": ["pick"]},
        next_rlt_obs=None,
        bootstrap_values=None,
        final_prev_values=torch.full((1, 1), 5.0),
        initial_transition=EnvTransition(dones=torch.zeros(1, 1, dtype=torch.bool)),
    )

    step = TrajectoryStep.from_parts(
        policy,
        env,
        rewards=env.transition.rewards,
        collect_prev_infos=True,
        collect_transitions=True,
        enable_rlt=False,
        include_final_value=True,
    )

    assert torch.equal(step.actions, torch.tensor([[1.0, 2.0, 7.0, 6.0]]))
    assert torch.equal(step.forward_inputs["action"], step.actions)
    assert "model_action" not in step.forward_inputs
    assert "task_descriptions" not in step.curr_obs
    assert "task_descriptions" not in step.next_obs
    assert torch.equal(step.final_prev_values, torch.full((1, 1), 5.0))


def test_trajectory_owns_step_materialization_splitting_and_batching():
    steps = [
        TrajectoryStep(
            actions=torch.tensor([[1.0], [2.0]]),
            rewards=torch.ones(2, 1),
            dones=torch.zeros(2, 1, dtype=torch.bool),
            initial_dones=torch.zeros(2, 1, dtype=torch.bool),
            forward_inputs={"action": torch.tensor([[1.0], [2.0]])},
            versions=torch.ones(2, 1),
        ),
        TrajectoryStep(
            actions=torch.tensor([[3.0], [4.0]]),
            rewards=torch.ones(2, 1),
            dones=torch.ones(2, 1, dtype=torch.bool),
            final_prev_values=torch.zeros(2, 1),
            forward_inputs={"action": torch.tensor([[3.0], [4.0]])},
            versions=torch.ones(2, 1),
        ),
    ]

    trajectory = Trajectory.from_steps(steps, max_episode_length=8)
    shards = trajectory.split(2)
    batch = Trajectory.to_batch(shards)

    assert trajectory.actions.shape == (2, 2, 1)
    assert trajectory.dones.shape == (3, 2, 1)
    assert [shard.actions.shape for shard in shards] == [(2, 1, 1)] * 2
    assert torch.equal(batch["actions"], trajectory.actions)
    assert torch.equal(batch["forward_inputs"]["action"], trajectory.actions)


def test_policy_input_methods_replace_legacy_routing_helpers():
    policy_input = PolicyInput(
        obs={"states": torch.arange(8).reshape(4, 2)},
        sources=[TrajectorySource(TrajectoryKey(1, 0, 0, 0, 0), 4)],
    )

    shards = policy_input.split([1, 3])
    merged = PolicyInput.merge(shards)

    assert torch.equal(merged.obs["states"], policy_input.obs["states"])


def test_accumulators_are_not_public_schema_api():
    import rlinf.data.schema as schema

    assert not hasattr(schema, "TrajectoryAccumulator")
    assert not hasattr(schema, "EmbodiedTrajectoryBuilder")
    assert not hasattr(schema, "EmbodiedLerobotTrajectoryBuilder")


def _config(**overrides):
    cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "auto_reset": True,
                    "ignore_terminations": False,
                    "max_episode_steps": 4,
                    "max_steps_per_rollout_epoch": 1,
                    "rollout_epoch": 1,
                    "total_num_envs": 1,
                }
            },
            "rollout": {
                "collect_prev_infos": True,
                "collect_transitions": False,
                "pipeline_stage_num": 1,
            },
            "actor": {
                "micro_batch_size": 1,
                "seed": 1,
                "model": {"action_dim": 1, "num_action_chunks": 1},
            },
            "algorithm": {
                "adv_type": "gae",
                "dagger": {"online_lerobot": {"enabled": False}},
                "gae_lambda": 1.0,
                "gamma": 0.5,
                "group_size": 1,
                "loss_type": "actor_critic",
                "normalize_advantages": False,
                "reward_type": "chunk_level",
                "shuffle_rollout": False,
            },
            "reward": {"env_reward_weight": 1.0, "reward_weight": 1.0},
            "runner": {
                "enable_decoupled_mode": False,
                "task_type": "embodied",
                "use_training_pipeline": False,
            },
        }
    )
    return OmegaConf.merge(cfg, overrides)


def _source(key: TrajectoryKey, size: int = 1, offset: int = 0):
    return TrajectorySource(key, size, offset)


def _policy(
    key: TrajectoryKey,
    value: float = 1.0,
    *,
    external: bool = False,
    **source_kwargs,
):
    value_tensor = torch.tensor([[value]])
    kwargs = (
        {"external_actions": value_tensor.unsqueeze(1)}
        if external
        else {
            "output": PolicyOutput(
                forward_inputs={"action": value_tensor},
                prev_logprobs=torch.zeros(1, 1),
                prev_values=torch.zeros(1, 1),
                versions=torch.full((1, 1), 7.0),
            )
        }
    )
    return PolicyPart(
        sources=[_source(key, **source_kwargs)],
        obs={"states": value_tensor},
        **kwargs,
    )


def _env_part(
    key: TrajectoryKey,
    *,
    episode_data=None,
    initial_transition=None,
    value: float = 2.0,
    **source_kwargs,
):
    value_tensor = torch.tensor([[value]])
    return EnvPart(
        sources=[_source(key, **source_kwargs)],
        transition=EnvTransition(
            rewards=torch.ones(1, 1),
            dones=torch.ones(1, 1, dtype=torch.bool),
            truncations=torch.ones(1, 1, dtype=torch.bool),
            terminations=torch.zeros(1, 1, dtype=torch.bool),
            episode_data=episode_data,
        ),
        next_obs={"states": value_tensor},
        next_rlt_obs={"states": value_tensor},
        bootstrap_values=torch.tensor([[4.0]]),
        final_prev_values=torch.tensor([[5.0]]),
        initial_transition=initial_transition,
    )


def _episode_data():
    return {
        "chunk_actions": torch.zeros(1, 1, 1),
        "obs_list": [{"states": torch.zeros(1, 1)}],
        "terminations": torch.zeros(1, 1, dtype=torch.bool),
        "truncations": torch.zeros(1, 1, dtype=torch.bool),
        "infos_list": [{}],
    }


def _make(cfg=None, **geometry):
    """Build the public collector with a deterministic cluster geometry."""
    shape = {
        "source_count": 1,
        "chunk_count": 1,
        "shards_per_source": 1,
        "actor_world_size": 1,
    }
    shape.update(geometry)
    collector = TrajectoryCollector()
    with patch.object(
        RolloutGeometry, "from_cfg", return_value=RolloutGeometry(**shape)
    ):
        collector.setup(
            ChannelContext(name="Actor", cfg=cfg if cfg is not None else _config())
        )
    return collector


def _collect(collector, *parts):
    outputs = []
    for part in parts:
        outputs.extend(collector.collect(part, DEFAULT_KEY))
    return outputs


def test_policy_part_requires_exactly_one_policy_payload():
    key = TrajectoryKey(0, 0, 0, 0, 0)
    common = {"sources": [_source(key)], "obs": {"states": torch.zeros(1, 1)}}

    with pytest.raises(ValueError, match="exactly one"):
        PolicyPart(**common)
    with pytest.raises(ValueError, match="exactly one"):
        PolicyPart(
            **common,
            output=PolicyOutput(forward_inputs={}),
            external_actions=torch.zeros(1, 1),
        )


def test_setup_requires_the_run_config():
    with pytest.raises(ValueError, match="needs the run config"):
        TrajectoryCollector().setup(ChannelContext(name="Actor", cfg=None))


@pytest.mark.parametrize(
    ("overrides", "mode", "dispatcher"),
    [
        ({}, TrajectoryMode.ROLLOUT, "least_loaded"),
        (
            {"runner": {"enable_decoupled_mode": False}},
            TrajectoryMode.ROLLOUT,
            "least_loaded",
        ),
        (
            {"runner": {"enable_decoupled_mode": True}},
            TrajectoryMode.ROLLOUT,
            "least_loaded",
        ),
        (
            {"runner": {"use_training_pipeline": True}},
            TrajectoryMode.PIPELINE,
            None,
        ),
        (
            {"algorithm": {"dagger": {"online_lerobot": {"enabled": True}}}},
            TrajectoryMode.LEROBOT,
            "least_loaded",
        ),
    ],
    ids=["sync", "async", "decoupled", "pipeline", "lerobot"],
)
def test_plan_is_the_single_mode_and_dispatcher_source(overrides, mode, dispatcher):
    cfg = _config(**overrides)

    assert TrajectoryPlan.mode_from_cfg(cfg) is mode
    assert select_trajectory_collector(cfg) is TrajectoryCollector
    assert select_trajectory_dispatcher(cfg) == dispatcher


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "runner": {
                "enable_decoupled_mode": True,
                "use_training_pipeline": True,
            }
        },
        {
            "runner": {"use_training_pipeline": True},
            "algorithm": {"dagger": {"online_lerobot": {"enabled": True}}},
        },
        {
            "runner": {"use_training_pipeline": True},
            "algorithm": {"adv_type": "opd"},
        },
    ],
    ids=["pipeline-decoupled", "pipeline-lerobot", "pipeline-opd"],
)
def test_plan_rejects_unsupported_mode_combinations(overrides):
    with pytest.raises(ValueError, match="does not support"):
        TrajectoryPlan.mode_from_cfg(_config(**overrides))


@pytest.mark.parametrize(
    ("decoupled", "part_order"),
    [
        (False, ("policy", "env")),
        (False, ("env", "policy")),
        (True, ("policy", "env")),
        (True, ("env", "policy")),
    ],
    ids=[
        "sync-policy-first",
        "sync-env-first",
        "decoupled-policy-first",
        "decoupled-env-first",
    ],
)
def test_rollout_output_matches_main_fields_for_sync_and_decoupled(
    decoupled, part_order
):
    """Assert the fields built by main's EnvWorker remain byte-for-byte equal."""
    cfg = _config(runner={"enable_decoupled_mode": decoupled})
    collector = _make(cfg)
    key = TrajectoryKey(3, 0, 0, 0, 0)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    parts = {
        "policy": _policy(key),
        "env": _env_part(key, initial_transition=initial),
    }

    [(_, trajectory)] = _collect(collector, *(parts[name] for name in part_order))

    # These are the exact values main appends in EnvWorker._run_interact_once:
    # one initial boundary, one policy step, and gamma * terminal bootstrap.
    assert torch.equal(trajectory.actions, torch.tensor([[[1.0]]]))
    assert torch.equal(trajectory.prev_logprobs, torch.tensor([[[0.0]]]))
    assert torch.equal(trajectory.prev_values, torch.tensor([[[0.0]], [[5.0]]]))
    assert torch.equal(trajectory.rewards, torch.tensor([[[3.0]]]))
    assert torch.equal(trajectory.dones, torch.tensor([[[False]], [[True]]]))
    assert torch.equal(trajectory.truncations, torch.tensor([[[False]], [[True]]]))
    assert torch.equal(trajectory.terminations, torch.tensor([[[False]], [[False]]]))
    assert torch.equal(trajectory.versions, torch.tensor([[[7.0]]]))
    assert torch.equal(trajectory.forward_inputs["action"], torch.tensor([[[1.0]]]))


@pytest.mark.parametrize("decoupled", [False, True], ids=["async", "decoupled"])
def test_rollout_sources_flush_independently_like_main_env_workers(decoupled):
    """A slow env source must not add a global barrier outside pipeline mode."""
    cfg = _config(
        env={"train": {"total_num_envs": 2}},
        runner={"enable_decoupled_mode": decoupled},
    )
    collector = _make(cfg, source_count=2)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )

    first_key = TrajectoryKey(4, 0, 0, 0, 0)
    first_outputs = _collect(
        collector,
        _policy(first_key),
        _env_part(first_key, initial_transition=initial),
    )
    assert len(first_outputs) == 1

    second_key = TrajectoryKey(4, 0, 1, 0, 0)
    second_outputs = _collect(
        collector,
        _env_part(second_key, initial_transition=initial),
        _policy(second_key),
    )
    assert len(second_outputs) == 1


def test_collector_joins_out_of_order_routed_fragments_and_initial_state():
    collector = _make(source_count=1, chunk_count=1, shards_per_source=1)
    collector._joiner._source_batch_size = 2
    key = TrajectoryKey(0, 0, 0, 0, 0)
    initial = EnvTransition(dones=torch.zeros(2, 1, dtype=torch.bool))
    policy_tail = _policy(key, 11.0, size=1, offset=1)
    policy_head = _policy(key, 10.0, size=1, offset=0)
    env = EnvPart(
        sources=[_source(key, size=2)],
        transition=EnvTransition(
            rewards=torch.ones(2, 1),
            dones=torch.ones(2, 1, dtype=torch.bool),
            truncations=torch.zeros(2, 1, dtype=torch.bool),
        ),
        next_obs={"states": torch.tensor([[20], [21]])},
        next_rlt_obs=None,
        bootstrap_values=None,
        final_prev_values=None,
        initial_transition=initial,
    )

    assert _collect(collector, policy_tail, policy_head) == []
    [(_, trajectory)] = _collect(collector, env)

    assert torch.equal(trajectory.actions, torch.tensor([[[10.0], [11.0]]]))
    assert torch.equal(trajectory.dones[0], initial.dones)


def test_collector_rejects_initial_state_after_chunk_zero():
    collector = _make()
    key = TrajectoryKey(0, 0, 0, 0, 1)

    _collect(collector, _policy(key))
    with pytest.raises(ValueError, match="Only chunk zero"):
        _collect(collector, _env_part(key, initial_transition=EnvTransition()))


def test_collector_requires_initial_state_for_chunk_zero():
    collector = _make()
    key = TrajectoryKey(0, 0, 0, 0, 0)

    _collect(collector, _policy(key))
    with pytest.raises(ValueError, match="missing its initial state"):
        _collect(collector, _env_part(key))


def test_collector_keeps_joined_parts_when_output_materialization_fails():
    collector = _make()
    key = TrajectoryKey(0, 0, 0, 0, 0)
    policy = _policy(key)
    env = _env_part(key, initial_transition=EnvTransition())
    original_emit = collector._output.emit
    collector._output.emit = Mock(side_effect=RuntimeError("invalid output"))

    _collect(collector, policy)
    with pytest.raises(RuntimeError, match="invalid output"):
        _collect(collector, env)

    collector._output.emit = original_emit
    assert len(_collect(collector, policy)) == 1


def test_rollout_collector_rejects_duplicate_completed_key():
    cfg = _config(env={"train": {"rollout_epoch": 2}})
    collector = _make(cfg)
    key = TrajectoryKey(3, 0, 0, 0, 0)

    assert (
        _collect(
            collector,
            _policy(key),
            _env_part(key, initial_transition=EnvTransition()),
        )
        == []
    )
    with pytest.raises(ValueError, match="duplicate trajectory event"):
        _collect(
            collector,
            _policy(key),
            _env_part(key, initial_transition=EnvTransition()),
        )


def test_rollout_materializes_each_epoch_boundary_and_final_value_in_order():
    collector = _make(_config(env={"train": {"rollout_epoch": 2}}))
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    first_key = TrajectoryKey(3, 0, 0, 0, 0)
    second_key = TrajectoryKey(3, 1, 0, 0, 0)

    assert (
        _collect(
            collector,
            _policy(first_key),
            _env_part(first_key, initial_transition=initial),
        )
        == []
    )
    [(_, trajectory)] = _collect(
        collector,
        _env_part(second_key, initial_transition=initial),
        _policy(second_key),
    )

    assert torch.equal(
        trajectory.dones,
        torch.tensor([[[False]], [[True]], [[False]], [[True]]]),
    )
    assert torch.equal(
        trajectory.prev_values,
        torch.tensor([[[0.0]], [[5.0]], [[0.0]], [[5.0]]]),
    )


def test_rollout_omits_policy_statistics_when_collection_is_disabled():
    collector = _make(_config(rollout={"collect_prev_infos": False}))
    key = TrajectoryKey(3, 0, 0, 0, 0)

    [(_, trajectory)] = _collect(
        collector,
        _policy(key),
        _env_part(key, initial_transition=EnvTransition()),
    )

    assert trajectory.prev_logprobs is None
    assert trajectory.prev_values is None


def test_history_reward_assignment_matches_main_across_chunks():
    cfg = _config(
        env={
            "train": {
                "auto_reset": False,
                "max_steps_per_rollout_epoch": 2,
            }
        },
        reward={
            "env_reward_weight": 1.0,
            "history_reward_assign": True,
            "reward_mode": "history_buffer",
            "reward_weight": 2.0,
        },
    )
    collector = _make(cfg, chunk_count=2)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    first_key = TrajectoryKey(3, 0, 0, 0, 0)
    second_key = TrajectoryKey(3, 0, 0, 0, 1)
    first_env = _env_part(first_key, initial_transition=initial)
    first_env.transition.dones = torch.zeros(1, 1, dtype=torch.bool)
    first_env.transition.truncations = torch.zeros(1, 1, dtype=torch.bool)
    second_env = _env_part(second_key)
    second_env.transition.reward_model_output = torch.tensor([[3.0]])
    second_env.transition.reward_assign_lengths = [2]

    assert _collect(collector, _policy(first_key), first_env) == []
    [(_, trajectory)] = _collect(
        collector,
        second_env,
        _policy(second_key),
    )

    # Current reward: 1 + 2 * 3. History assignment adds 2 * 3 to step 0.
    assert torch.equal(trajectory.rewards, torch.tensor([[[7.0]], [[7.0]]]))


@pytest.mark.parametrize("loss_type", ["rlt_ac", "rlt_td3"])
def test_rlt_output_matches_main_transition_and_intervention_behavior(loss_type):
    collector = _make(_config(algorithm={"loss_type": loss_type}))
    key = TrajectoryKey(3, 0, 0, 0, 0)
    current_ref_chunk = torch.tensor([[1.0, 2.0]])
    intervened_chunk = torch.tensor([[9.0, 8.0]])
    policy = PolicyPart(
        sources=[_source(key)],
        obs={"states": torch.zeros(1, 1)},
        output=PolicyOutput(
            forward_inputs={
                "action": current_ref_chunk,
                "z_rl": torch.tensor([[3.0, 4.0]]),
                "proprio": torch.tensor([[5.0, 6.0, 7.0]]),
                "ref_chunk": current_ref_chunk,
            },
            prev_logprobs=torch.zeros(1, 1),
            prev_values=torch.zeros(1, 1),
        ),
    )
    env = EnvPart(
        sources=[_source(key)],
        transition=EnvTransition(
            rewards=torch.ones(1, 1),
            dones=torch.zeros(1, 1, dtype=torch.bool),
            truncations=torch.zeros(1, 1, dtype=torch.bool),
            intervene_actions=intervened_chunk,
            intervene_flags=torch.ones(1, 1, dtype=torch.bool),
        ),
        next_obs={"states": torch.ones(1, 1)},
        next_rlt_obs={
            "z_rl": torch.tensor([[13.0, 14.0]]),
            "proprio": torch.tensor([[15.0, 16.0, 17.0]]),
            "ref_chunk": torch.tensor([[11.0, 12.0]]),
        },
        bootstrap_values=None,
        final_prev_values=None,
        initial_transition=EnvTransition(),
    )

    [(_, trajectory)] = _collect(collector, env, policy)

    assert set(trajectory.curr_obs) == {"z_rl", "proprio", "ref_chunk"}
    assert set(trajectory.next_obs) == {"z_rl", "proprio", "ref_chunk"}
    assert torch.equal(trajectory.curr_obs["ref_chunk"][0], intervened_chunk)
    assert torch.equal(
        trajectory.next_obs["ref_chunk"][0], torch.tensor([[11.0, 12.0]])
    )


def test_pipeline_output_contains_main_actor_training_fields():
    collector = _make(
        _config(runner={"use_training_pipeline": True}),
        actor_world_size=1,
    )
    key = TrajectoryKey(2, 0, 0, 0, 0)
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )

    [(queue_key, batch)] = _collect(
        collector,
        _policy(key),
        _env_part(key, initial_transition=initial),
    )

    assert queue_key == "0_0_pipeline_actor"
    assert {
        "actions",
        "advantages",
        "dones",
        "prev_logprobs",
        "prev_values",
        "returns",
        "rewards",
    } <= batch.keys()
    assert batch["actions"].shape[0] == 1
    assert batch["advantages"].shape == batch["returns"].shape


def test_pipeline_waits_for_all_sources_and_routes_each_actor_like_main():
    cfg = _config(
        env={"train": {"total_num_envs": 2}},
        runner={"use_training_pipeline": True},
    )
    collector = _make(
        cfg,
        source_count=2,
        actor_world_size=2,
    )
    initial = EnvTransition(
        dones=torch.zeros(1, 1, dtype=torch.bool),
        truncations=torch.zeros(1, 1, dtype=torch.bool),
        terminations=torch.zeros(1, 1, dtype=torch.bool),
    )
    first_key = TrajectoryKey(2, 0, 0, 0, 0)
    second_key = TrajectoryKey(2, 0, 1, 0, 0)

    assert (
        _collect(
            collector,
            _policy(first_key),
            _env_part(first_key, initial_transition=initial),
        )
        == []
    )
    outputs = _collect(
        collector,
        _env_part(second_key, initial_transition=initial),
        _policy(second_key),
    )

    assert {queue_key for queue_key, _ in outputs} == {
        "0_0_pipeline_actor",
        "1_1_pipeline_actor",
    }
    assert all("advantages" in batch for _, batch in outputs)


def test_pipeline_flushes_each_epoch_to_the_actor_specific_key():
    collector = _make(
        _config(runner={"use_training_pipeline": True}),
        actor_world_size=1,
    )
    collector._output._prepare_pipeline_batch = lambda trajectory: {"value": trajectory}
    collector._output._pipeline_micro_batches = lambda batch, actor_rank: [batch]

    for epoch_id in (0, 1):
        key = TrajectoryKey(2, epoch_id, 0, 0, 0)
        outputs = _collect(
            collector,
            _env_part(key, initial_transition=EnvTransition()),
            _policy(key),
        )
        assert outputs[0][0] == "0_0_pipeline_actor"
        assert (2, epoch_id) not in collector._output._accumulators


def test_online_lerobot_accepts_external_policy_actions_and_emits_shards():
    cfg = _config(
        algorithm={"dagger": {"online_lerobot": {"enabled": True}}},
    )
    collector = _make(cfg, shards_per_source=2)
    key = TrajectoryKey(3, 0, 0, 0, 0)

    outputs = _collect(
        collector,
        _policy(key, external=True),
        _env_part(
            key,
            episode_data=_episode_data(),
            initial_transition=EnvTransition(),
        ),
    )

    assert len(outputs) == 2
    assert all(key == DEFAULT_KEY for key, _ in outputs)


def test_online_lerobot_accumulator_records_external_intervened_action():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=1,
        action_dim=2,
    )

    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0, 2.0]]]),
        obs_list=[
            {
                "states": torch.tensor([[0.0, 0.0]]),
                "task_descriptions": ["pick"],
            }
        ],
        terminations=torch.tensor([[True]]),
        truncations=torch.tensor([[False]]),
        infos_list=[
            {
                "intervene_action": torch.tensor([[9.0, 8.0]]),
                "intervene_flag": torch.tensor([True]),
            }
        ],
    )

    [episode] = accumulator.drain_episodes()
    assert torch.equal(
        torch.from_numpy(episode[0]["actions"]), torch.tensor([9.0, 8.0])
    )


def test_online_lerobot_accumulator_preserves_auto_reset_observation():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=1,
        action_dim=1,
    )

    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0]]]),
        obs_list=[{"states": torch.tensor([[100.0]])}],
        terminations=torch.tensor([[True]]),
        truncations=torch.tensor([[False]]),
        infos_list=[
            {
                "final_observation": {"states": torch.tensor([[10.0]])},
                "final_info": {"success_once": torch.tensor([True])},
            }
        ],
    )
    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[2.0]]]),
        obs_list=[{"states": torch.tensor([[20.0]])}],
        terminations=torch.tensor([[True]]),
        truncations=torch.tensor([[False]]),
        infos_list=[{}],
    )

    first, second = accumulator.drain_episodes()
    assert first[0]["state"].item() == 10.0
    assert first[0]["is_success"].item()
    assert second[0]["state"].item() == 100.0


def test_online_lerobot_accumulator_filters_unsuccessful_episodes():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        only_success=True,
        num_action_chunks=1,
        action_dim=1,
    )

    for terminated, truncated, success in (
        (True, False, False),
        (False, True, True),
        (True, False, True),
    ):
        accumulator.append_chunk_episode_data(
            policy_output=None,
            chunk_actions=torch.tensor([[[1.0]]]),
            obs_list=[{"states": torch.tensor([[1.0]])}],
            terminations=torch.tensor([[terminated]]),
            truncations=torch.tensor([[truncated]]),
            infos_list=[{"success_once": torch.tensor([success])}],
        )

    [episode] = accumulator.drain_episodes()
    assert episode[-1]["is_success"].item()
    assert episode[-1]["done"].item()


def test_online_lerobot_accumulator_expands_vectorized_action_chunks():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=2,
        num_action_chunks=2,
        action_dim=1,
    )

    accumulator.append_chunk_episode_data(
        policy_output=None,
        chunk_actions=torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        obs_list=[
            {"states": torch.tensor([[10.0], [30.0]])},
            {"states": torch.tensor([[20.0], [40.0]])},
        ],
        terminations=torch.tensor([[False, True], [False, True]]),
        truncations=torch.zeros(2, 2, dtype=torch.bool),
        infos_list=[{}, {}],
    )

    episodes = accumulator.drain_episodes()
    assert len(episodes) == 2
    assert [frame["actions"].item() for frame in episodes[0]] == [1.0, 2.0]
    assert [frame["actions"].item() for frame in episodes[1]] == [3.0, 4.0]


def test_online_lerobot_accumulator_uses_policy_intervention_metadata():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=2,
        action_dim=1,
    )
    policy_output = PolicyOutput(
        forward_inputs={"action": torch.tensor([[[9.0], [8.0]]])},
        intervene_flags=torch.tensor([[False, True]]),
    )
    chunk_actions = torch.tensor([[[1.0], [2.0]]])

    accumulator.append_chunk_episode_data(
        policy_output=policy_output,
        chunk_actions=chunk_actions,
        obs_list=[
            {"states": torch.tensor([[10.0]])},
            {"states": torch.tensor([[20.0]])},
        ],
        terminations=torch.tensor([[False, True]]),
        truncations=torch.zeros(1, 2, dtype=torch.bool),
        infos_list=[{}, {}],
    )

    [episode] = accumulator.drain_episodes()
    assert [frame["actions"].item() for frame in episode] == [1.0, 8.0]
    assert [frame["intervene_flag"].item() for frame in episode] == [False, True]


def test_online_lerobot_accumulator_applies_recording_controls():
    accumulator = LeRobotEpisodeAccumulator(
        num_envs=1,
        num_action_chunks=1,
        action_dim=1,
    )

    for state, action, info, terminated in (
        (10.0, 1.0, {"record_reset": True}, False),
        (20.0, 2.0, {"pre_record": True}, False),
        (30.0, 3.0, {"segment_advance": True}, True),
    ):
        accumulator.append_chunk_episode_data(
            policy_output=None,
            chunk_actions=torch.tensor([[[action]]]),
            obs_list=[{"states": torch.tensor([[state]])}],
            terminations=torch.tensor([[terminated]]),
            truncations=torch.tensor([[False]]),
            infos_list=[info],
        )

    [episode] = accumulator.drain_episodes()
    assert len(episode) == 1
    assert episode[0]["state"].item() == 10.0
    assert episode[0]["actions"].item() == 3.0
    assert episode[0]["segment_id"].item() == 1


def test_lerobot_frame_owns_observation_and_image_conversion():
    frame = LeRobotFrame.from_step(
        observation={
            "main_images": torch.full((2, 2, 3), 0.5),
            "wrist_images": torch.zeros(2, 2, 2, 3),
            "states": torch.tensor([1.0, 2.0]),
            "task_descriptions": ["pick"],
        },
        action=torch.tensor([1.0, 2.0]).numpy(),
        info={
            "intervene_action": torch.tensor([9.0, 8.0]),
            "intervene_flag": torch.tensor(True),
            "success_once": torch.tensor(True),
        },
        segment_id=3,
        action_dim=2,
    )

    assert frame is not None
    output = frame.to_dict(episode_success=True, done=True)
    assert output["actions"].tolist() == [9.0, 8.0]
    assert output["image"].dtype.name == "uint8"
    assert {"wrist_image-0", "wrist_image-1"} <= output.keys()
    assert output["task"] == "pick"
    assert output["segment_id"].item() == 3


def test_offline_lerobot_export_reuses_canonical_frame_conversion():
    collector = object.__new__(CollectEpisode)
    collector.num_envs = 1
    buffer = {
        "observations": [
            {
                "main_images": torch.full((2, 2, 3), 0.5),
                "states": torch.tensor([1.0, 2.0]),
                "task_descriptions": ["pick"],
            }
        ],
        "actions": [torch.tensor([1.0, 2.0])],
        "terminated": [True],
        "infos": [
            {},
            {
                "intervene_action": torch.tensor([9.0, 8.0]),
                "intervene_flag": torch.tensor([True]),
            },
        ],
        "segment_ids": [4],
    }

    [frame] = collector._buffer_to_lerobot_ep(buffer, env_idx=0, is_success=True)
    assert frame["actions"].tolist() == [9.0, 8.0]
    assert frame["image"].dtype.name == "uint8"
    assert frame["task"] == "pick"
    assert frame["segment_id"].item() == 4
    assert frame["done"].item()


def _policy_input() -> PolicyInput:
    return PolicyInput(
        obs={"states": torch.zeros(16, 4)},
    )


def test_policy_input_compression_preserves_attached_env_parts(monkeypatch):
    """Trajectory requests remain lossless across split/compress/merge routing."""
    monkeypatch.setattr(
        obs_compression,
        "_get_backend",
        lambda _codec: (
            lambda raw, _level: raw,
            lambda raw: raw,
        ),
    )
    key = TrajectoryKey(0, 0, 0, 0, 0)
    source = TrajectorySource(key=key, size=4)
    current_images = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)
    next_images = torch.randint(0, 256, (4, 8, 8, 3), dtype=torch.uint8)
    policy_input = PolicyInput(
        obs={"main_images": current_images},
        sources=[source],
        env_parts=[
            EnvPart(
                sources=[source],
                transition=EnvTransition(rewards=torch.ones(4, 2)),
                next_obs={"main_images": next_images},
            )
        ],
    )
    env_worker = object.__new__(EnvWorker)
    env_worker.obs_compression_cfg = {
        "enable": True,
        "codec": "lz4",
        "level": 1,
        "xor_delta": True,
    }

    shards = env_worker._split_and_compress_policy_input(policy_input, [1, 3])

    assert is_compressed_image(shards[0].obs["main_images"])
    assert is_compressed_image(shards[0].env_parts[0].next_obs["main_images"])
    merged = MultiStepRolloutWorker._merge_policy_inputs(shards)
    assert torch.equal(merged.obs["main_images"], current_images)
    assert torch.equal(
        torch.cat([part.next_obs["main_images"] for part in merged.env_parts], dim=0),
        next_images,
    )


def test_sync_and_async_workers_share_the_part_channel_contract():
    """Only Rollout publishes parts; Env communicates exclusively with Rollout."""
    assert "actor_channel" not in inspect.signature(EnvWorker.interact).parameters
    assert "actor_channel" not in inspect.signature(AsyncEnvWorker.interact).parameters
    assert (
        "actor_channel" in inspect.signature(MultiStepRolloutWorker.generate).parameters
    )
    assert (
        "actor_channel"
        in inspect.signature(AsyncMultiStepRolloutWorker.generate).parameters
    )


def test_decoupled_evaluation_service_is_reused_and_cancelled_on_stop():
    async def run():
        worker = object.__new__(AsyncMultiStepRolloutWorker)
        worker.env_decoupled_mode = True
        worker._generate_task = None
        worker._evaluate_task = None
        blocker = asyncio.Event()
        calls = 0

        async def serve(_input_channel, _output_channel):
            nonlocal calls
            calls += 1
            await blocker.wait()

        worker._run_evaluate_service = serve
        await worker.ensure_evaluate_service(Mock(), Mock())
        first_task = worker._evaluate_task
        await worker.ensure_evaluate_service(Mock(), Mock())

        assert worker._evaluate_task is first_task
        assert calls == 1

        worker.stop()
        await asyncio.sleep(0)
        assert first_task.cancelled()

    asyncio.run(run())


def test_async_runner_reuses_decoupled_evaluation_service_across_validations():
    runner = object.__new__(AsyncEmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {"enable_decoupled_mode": True}})
    runner.env_channel = Mock()
    runner.rollout_channel = Mock()
    runner.rollout = Mock()
    runner.rollout.ensure_evaluate_service.return_value = Mock(
        wait=Mock(return_value=None)
    )
    runner.env = Mock()
    runner.env.evaluate.return_value = Mock(wait=Mock(return_value=[{}]))

    with patch(
        "rlinf.runners.async_embodied_runner.compute_evaluate_metrics",
        return_value={},
    ):
        runner.evaluate()
        runner.evaluate()

    assert runner.rollout.ensure_evaluate_service.call_count == 2
    runner.rollout.evaluate.assert_not_called()


def test_decoupled_policy_route_round_trip():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.env_decoupled_mode = True
    rollout.cfg = SimpleNamespace(env=SimpleNamespace(group_name="EnvGroup"))
    rollout.train_batch_size = 32
    rollout.rollout_queue_size = 0
    rollout.batch_router = {"policy": ["stale"]}
    rollout.recv_from_and_record_batch_routes_with_timeout = AsyncMock(
        return_value=(_policy_input(), [8, 8])
    )
    rollout.send_to_recorded_batch_routes = Mock()

    policy_input, split_sizes = asyncio.run(
        rollout._receive_policy_input(None, "policy", 0)
    )
    source = torch.zeros(16, 4, requires_grad=True)
    output = source * 2
    rollout._send_actions(None, output, 0, split_sizes)

    assert policy_input.obs["states"].shape == (16, 4)
    rollout.recv_from_and_record_batch_routes_with_timeout.assert_awaited_once_with(
        group_name="EnvGroup",
        channel=None,
        tag="policy",
        batch_size=32,
        merge_fn=rollout._merge_policy_inputs,
        infer_batch_size_fn=rollout._infer_policy_input_batch_size,
        timeout_time=0.02,
        recv_queue_size=0,
    )
    sent_output = rollout.send_to_recorded_batch_routes.call_args.kwargs["data"]
    assert torch.equal(sent_output, output)
    assert not sent_output.requires_grad
    assert sent_output.grad_fn is None
    rollout.send_to_recorded_batch_routes.assert_called_once_with(
        group_name="EnvGroup",
        channel=None,
        data=sent_output,
        tag="policy",
        split_fn=rollout._split_actions,
        split_sizes=[8, 8],
    )


def test_env_uses_mode_qualified_decoupled_response_tag():
    env = object.__new__(EnvWorker)
    env.cfg = SimpleNamespace(rollout=SimpleNamespace(group_name="RolloutGroup"))
    env.env_decoupled_mode = True
    env.train_batch_size = 32
    env.recv_from = Mock(return_value=torch.zeros(8, 4))

    env._recv_actions(None, stage_id=0)

    env.recv_from.assert_called_once_with(
        group_name="RolloutGroup",
        channel=None,
        tag="train_policy",
        route_key=None,
        batch_size=32,
        infer_batch_size_fn=env._infer_action_batch_size,
        decoupled_mode=True,
    )


def test_smooth_intervention_builds_external_input_without_policy_history():
    controller = SmoothInterveneController(
        stage_num=1,
        num_envs_per_stage=1,
        num_action_chunks=2,
        action_dim=3,
        enabled=True,
    )
    env = SimpleNamespace(get_hold_actions=lambda fallback=None: [[1.0, 2.0, 3.0]])

    policy_input = controller.build_external_policy_input(
        0, env=env, obs={"states": torch.zeros(1, 4)}
    )

    assert isinstance(policy_input, PolicyInput)
    assert not policy_input.requires_inference
    assert torch.equal(
        policy_input.external_actions,
        torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]),
    )


def test_smooth_intervention_requires_online_lerobot_dagger():
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"num_action_chunks": 2, "action_dim": 3}},
            "algorithm": {
                "loss_type": "ppo",
                "dagger": {"online_lerobot": {"enabled": True}},
            },
            "env": {
                "train": {
                    "smooth_intervene": True,
                    "env_type": "realworld",
                    "use_pico": True,
                    "use_spacemouse": False,
                }
            },
        }
    )

    with pytest.raises(ValueError, match="loss_type=embodied_dagger"):
        SmoothInterveneController.from_cfg(
            cfg,
            stage_num=1,
            enable_train=True,
            train_num_envs_per_stage=1,
        )


def test_env_sends_external_request_after_intervention_continues():
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 2
    env.enable_online_lerobot = True
    env.env_decoupled_mode = False
    env.env_list = [
        SimpleNamespace(get_hold_actions=lambda fallback=None: [[1.0, 2.0]])
    ]
    env.smooth_intervene = SmoothInterveneController(1, 1, 2, 2, enabled=True)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    env_output = EnvOutput(
        obs={"states": torch.zeros(1, 4)},
        transition=EnvTransition(
            dones=torch.tensor([[False, False]]),
            intervene_flags=torch.tensor([[False, True]]),
        ),
    )

    env._publish_step(Mock(), env_output, EnvTransition(), None, {}, 0, 0, 0)

    policy_input = env._send_policy_input.call_args.args[1]
    assert not policy_input.requires_inference
    assert policy_input.env_parts[0] is not None
    assert policy_input.env_parts[0].next_obs is None
    assert torch.equal(policy_input.obs["states"], env_output.obs["states"])
    assert torch.equal(
        policy_input.external_actions,
        torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]),
    )


def _publish_step_env(**overrides):
    """Build the minimal EnvWorker used by _publish_step tests."""
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 2
    env.enable_online_lerobot = False
    env.env_decoupled_mode = False
    env.collect_final_values = True
    env.smooth_intervene = SmoothInterveneController(1, 1, 1, 4)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    for name, value in overrides.items():
        setattr(env, name, value)
    return env


def _terminal_env_output(reset_obs, terminal_obs, **transition):
    flags = {"dones": torch.ones(1, 1, dtype=torch.bool), **transition}
    return EnvOutput(
        obs={"states": reset_obs},
        final_obs={"states": terminal_obs},
        transition=EnvTransition(**flags),
    )


def test_env_sends_terminal_observation_only_as_a_boundary_override():
    env = _publish_step_env()
    reset_obs = torch.zeros(1, 4)
    terminal_obs = torch.ones(1, 4)
    env_output = _terminal_env_output(
        reset_obs,
        terminal_obs,
    )

    env._publish_step(Mock(), env_output, EnvTransition(), None, None, 0, 0, 0)

    policy_input = env._send_policy_input.call_args.args[1]
    assert torch.equal(policy_input.obs["states"], reset_obs)
    assert torch.equal(policy_input.env_parts[0].next_obs["states"], terminal_obs)
    assert policy_input.env_parts[0].requires_inference


def test_rollout_completes_each_epoch_through_policy_inputs():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.n_train_chunk_steps = 2
    rollout.num_pipeline_stages = 1
    rollout.env_decoupled_mode = False
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout.hf_model = SimpleNamespace(value_head=object())
    rollout.update_dagger_beta = Mock()
    rollout._send_actions = Mock()
    rollout._build_policy_output = Mock(
        side_effect=[
            PolicyOutput(
                forward_inputs={"states": torch.tensor([[1]])},
            ),
            PolicyOutput(
                forward_inputs={"states": torch.tensor([[2]])},
            ),
        ]
    )
    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    completion0 = EnvPart(
        sources=[TrajectorySource(key0, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.tensor([[2]])},
        requires_inference=False,
        initial_transition=EnvTransition(dones=torch.zeros(1, 1, dtype=torch.bool)),
    )
    completion1 = EnvPart(
        sources=[TrajectorySource(key1, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.tensor([[3]])},
        requires_inference=True,
    )
    rollout._receive_policy_input = AsyncMock(
        side_effect=[
            (
                PolicyInput(
                    obs={"states": torch.tensor([[1]])},
                    sources=[TrajectorySource(key0, 1)],
                    env_parts=[None],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.tensor([[2]])},
                    sources=[TrajectorySource(key1, 1)],
                    env_parts=[completion0],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.tensor([[3]])},
                    env_parts=[completion1],
                    request_sizes=[1],
                    is_last=True,
                ),
                None,
            ),
        ]
    )
    rollout._predict_rollout_actions = Mock(
        side_effect=[
            (torch.zeros(1, 1), {"forward_inputs": {"states": torch.tensor([[1]])}}),
            (torch.zeros(1, 1), {"forward_inputs": {"states": torch.tensor([[2]])}}),
            (
                torch.zeros(1, 1),
                {
                    "forward_inputs": {"states": torch.tensor([[3]])},
                    "prev_values": torch.tensor([[4.0, 5.0]]),
                },
            ),
        ]
    )
    actor_channel = Mock()
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__

    asyncio.run(
        generate_one_epoch(
            rollout,
            input_channel=Mock(),
            output_channel=Mock(),
            actor_channel=actor_channel,
        )
    )

    parts = [call.args[0] for call in actor_channel.put.call_args_list]
    assert sum(isinstance(part, PolicyPart) for part in parts) == 2
    env_parts = [part for part in parts if isinstance(part, EnvPart)]
    assert len(env_parts) == 2
    assert env_parts[0].initial_transition is completion0.initial_transition
    assert env_parts[0].next_rlt_obs is None
    assert env_parts[0].next_obs is None
    assert torch.equal(env_parts[1].bootstrap_values, torch.tensor([[4.0]]))
    assert torch.equal(env_parts[1].final_prev_values, torch.tensor([[4.0, 5.0]]))
    assert rollout._predict_rollout_actions.call_count == 3


def test_rollout_routes_external_input_without_model_inference():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.n_train_chunk_steps = 2
    rollout.num_pipeline_stages = 1
    rollout.env_decoupled_mode = False
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout.hf_model = SimpleNamespace(value_head=object())
    rollout.update_dagger_beta = Mock()
    rollout._send_actions = Mock()
    rollout._build_policy_output = Mock(
        return_value=PolicyOutput(
            forward_inputs={"action": torch.ones(1, 6)},
        )
    )
    key0 = TrajectoryKey(0, 0, 0, 0, 0)
    key1 = TrajectoryKey(0, 0, 0, 0, 1)
    completion0 = EnvPart(
        sources=[TrajectorySource(key0, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.ones(1, 4)},
        requires_inference=False,
    )
    completion1 = EnvPart(
        sources=[TrajectorySource(key1, 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.ones(1, 4)},
        requires_inference=False,
    )
    dummy_actions = torch.full((1, 2, 3), 2.0)
    rollout._receive_policy_input = AsyncMock(
        side_effect=[
            (
                PolicyInput(
                    obs={"states": torch.zeros(1, 4)},
                    sources=[TrajectorySource(key0, 1)],
                    env_parts=[None],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.ones(1, 4)},
                    external_actions=dummy_actions,
                    sources=[TrajectorySource(key1, 1)],
                    env_parts=[completion0],
                    request_sizes=[1],
                ),
                None,
            ),
            (
                PolicyInput(
                    obs={"states": torch.ones(1, 4)},
                    env_parts=[completion1],
                    request_sizes=[1],
                    is_last=True,
                ),
                None,
            ),
        ]
    )
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.ones(1, 2, 3),
            {"forward_inputs": {"action": torch.ones(1, 6)}},
        )
    )
    actor_channel = Mock()
    generate_one_epoch = MultiStepRolloutWorker.generate_one_epoch
    while hasattr(generate_one_epoch, "__wrapped__"):
        generate_one_epoch = generate_one_epoch.__wrapped__

    asyncio.run(generate_one_epoch(rollout, Mock(), Mock(), actor_channel))

    parts = [call.args[0] for call in actor_channel.put.call_args_list]
    inferred_parts = [
        part for part in parts if isinstance(part, PolicyPart) and part.inferred
    ]
    assert len(inferred_parts) == 1
    external_part = next(
        part for part in parts if isinstance(part, PolicyPart) and not part.inferred
    )
    assert torch.equal(external_part.external_actions, dummy_actions)
    assert rollout._predict_rollout_actions.call_count == 1
    assert torch.equal(rollout._send_actions.call_args_list[1].args[1], dummy_actions)


def test_decoupled_final_completion_uses_next_bootstrap():
    env = object.__new__(EnvWorker)
    env._trajectory_step = 0
    env._rank = 0
    env.stage_num = 1
    env.train_num_envs_per_stage = 1
    env.n_train_chunk_steps = 1
    env.enable_online_lerobot = False
    env.env_decoupled_mode = True
    env.smooth_intervene = SmoothInterveneController(1, 1, 1, 4)
    env._build_env_transition = Mock(return_value=EnvTransition())
    env._send_policy_input = Mock()
    env_output = EnvOutput(obs={"states": torch.zeros(1, 4)})

    completion = env._publish_step(
        rollout_channel=Mock(),
        env_output=env_output,
        initial_transition=EnvTransition(),
        reward_model_output=None,
        chunk_step_data=None,
        epoch_id=0,
        chunk_id=0,
        stage_id=0,
    )

    assert completion is not None
    assert completion.initial_transition is not None
    assert torch.equal(completion.next_obs["states"], env_output.obs["states"])
    env._send_policy_input.assert_not_called()

    env._send_train_bootstrap(
        rollout_channel=Mock(),
        env_outputs=[env_output],
        step_id=1,
        epoch_id=0,
        previous_env_parts={0: completion},
    )

    policy_input = env._send_policy_input.call_args.args[1]
    assert policy_input.env_parts == [completion]


def test_env_closes_reward_stream_with_final_chunk():
    env = object.__new__(EnvWorker)
    env.rollout_epoch = 2
    env.stage_num = 1
    env.n_train_chunk_steps = 2
    env._prefetched_train_bootstrap = None
    env._trajectory_step = 0
    env._rank = 0
    env.env_list = [SimpleNamespace()]
    env.use_training_pipeline = False
    env.env_decoupled_mode = False
    env.enable_online_lerobot = False
    env.cfg = SimpleNamespace(
        env=SimpleNamespace(
            train=SimpleNamespace(auto_reset=True, ignore_terminations=False)
        )
    )
    env._bootstrap_and_send_train = Mock(
        return_value=[EnvOutput(obs={"states": torch.zeros(1, 4)})]
    )
    env._maybe_wait_env_delay = AsyncMock()
    env._recv_actions = Mock(return_value=torch.zeros(1, 4))
    env.smooth_intervene = SmoothInterveneController(1, 1, 2, 4)
    env.env_interact_step = Mock(
        return_value=(
            EnvOutput(
                obs={"states": torch.zeros(1, 4)},
                transition=EnvTransition(rewards=torch.zeros(1, 1)),
            ),
            {},
            {},
        )
    )
    env.get_reward_model_output = Mock(return_value=torch.zeros(1, 1))
    env._publish_step = Mock()
    env.record_env_metrics = Mock()
    env.store_last_obs_and_intervened_info = Mock()
    env.finish_rollout = Mock()
    asyncio.run(
        EnvWorker._run_interact_once.__wrapped__(
            env,
            input_channel=Mock(),
            rollout_channel=Mock(),
            reward_channel=Mock(),
            cooperative_yield=False,
        )
    )

    assert env.get_reward_model_output.call_count == 4
    assert [
        call.kwargs["last_run"] for call in env.get_reward_model_output.call_args_list
    ] == [False, False, False, True]


def test_smooth_intervention_holds_at_the_last_commanded_step():
    controller = SmoothInterveneController(
        stage_num=1,
        num_envs_per_stage=1,
        num_action_chunks=2,
        action_dim=3,
        enabled=True,
    )
    seen = []

    def get_hold_actions(fallback=None):
        seen.append(fallback)
        return [[1.0, 2.0, 3.0]]

    env = SimpleNamespace(get_hold_actions=get_hold_actions)
    controller.remember_actions(0, torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]))

    controller.build_external_policy_input(
        0, env=env, obs={"states": torch.zeros(1, 4)}
    )

    # The wrapper receives the final step of the chunk that was just executed,
    # so it can hold that pose instead of snapping back to its own default.
    assert len(seen) == 1
    assert seen[0].tolist() == [[4.0, 5.0, 6.0]]


def test_rollout_runs_requested_terminal_inference_without_a_value_head():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = False
    rollout.collect_transitions = False
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.zeros(1, 1),
            {
                "prev_values": torch.zeros(1, 1),
                "forward_inputs": {"states": torch.ones(1, 1)},
            },
        )
    )
    actor_channel = Mock()
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.zeros(1, 4)},
        requires_inference=True,
    )

    rollout._publish_env_part(completion, None, None, actor_channel)

    rollout._predict_rollout_actions.assert_called_once_with(completion.next_obs)
    part = actor_channel.put.call_args.args[0]
    assert torch.equal(part.bootstrap_values, torch.zeros(1, 1))
    assert torch.equal(part.final_prev_values, torch.zeros(1, 1))
    assert part.next_rlt_obs is None
    assert part.next_obs is None


def test_rollout_reuses_policy_obs_for_transition_next_obs():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = False
    rollout.collect_transitions = True
    actor_channel = Mock()
    policy_obs = {"states": torch.ones(1, 4)}
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
    )

    rollout._publish_env_part(
        completion,
        policy_obs,
        {"unused": torch.ones(1, 1)},
        actor_channel,
    )

    part = actor_channel.put.call_args.args[0]
    assert part.next_obs is policy_obs
    assert part.next_rlt_obs is None


def test_rollout_runs_terminal_inference_for_rlt_without_a_value_head():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = True
    rollout.collect_transitions = True
    rollout.hf_model = SimpleNamespace()
    rollout._predict_rollout_actions = Mock(
        return_value=(
            torch.zeros(1, 1),
            {
                "forward_inputs": {
                    "rlt_transition_z_rl": torch.ones(1, 2),
                    "rlt_transition_proprio": torch.ones(1, 3),
                    "rlt_transition_ref_chunk": torch.ones(1, 4),
                }
            },
        )
    )
    actor_channel = Mock()
    completion = EnvPart(
        sources=[TrajectorySource(TrajectoryKey(0, 0, 0, 0, 0), 1)],
        transition=EnvTransition(),
        next_obs={"states": torch.zeros(1, 4)},
        requires_inference=True,
    )

    rollout._publish_env_part(completion, None, None, actor_channel)

    rollout._predict_rollout_actions.assert_called_once()
    part = actor_channel.put.call_args.args[0]
    assert torch.equal(part.next_rlt_obs["z_rl"], torch.ones(1, 2))


def test_rollout_keeps_next_forward_inputs_only_for_rlt():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_rlt = True
    rollout.collect_transitions = True
    actor_channel = Mock()
    key = TrajectoryKey(0, 0, 0, 0, 0)
    policy_input = PolicyInput(
        obs={"states": torch.zeros(1, 4)},
        env_parts=[
            EnvPart(
                sources=[TrajectorySource(key, 1)],
                transition=EnvTransition(),
            )
        ],
        request_sizes=[1],
    )
    forward_inputs = {
        "rlt_transition_z_rl": torch.ones(1, 2),
        "rlt_transition_proprio": torch.ones(1, 3),
        "rlt_transition_ref_chunk": torch.ones(1, 4),
        "unrelated": torch.ones(1, 8),
    }

    rollout._publish_env_parts(policy_input, forward_inputs, actor_channel)

    part = actor_channel.put.call_args.args[0]
    assert part.next_obs is None
    assert torch.equal(
        part.next_rlt_obs["z_rl"],
        forward_inputs["rlt_transition_z_rl"],
    )
    assert set(part.next_rlt_obs) == {"z_rl", "proprio", "ref_chunk"}


def test_vlm_trend_batch_video_metadata_stays_nested_per_sample():
    pytest.importorskip("transformers.video_utils")
    from transformers.video_utils import VideoMetadata

    from rlinf.data.datasets.vlm.vlm_trend_reward import VLMTrendRewardSFTDataset

    captured = {}

    class _Processor:
        video_token = "<|video_pad|>"

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            captured["videos_kwargs"] = kwargs["videos_kwargs"]
            batch = len(kwargs["text"])
            return {
                "input_ids": torch.zeros(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
            }

    VLMTrendRewardSFTDataset.process_inputs(
        processor=_Processor(),
        system_prompt=None,
        use_chat_template=True,
        prompt_texts=[["task a"], ["task b"]],
        videos=[
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
        ],
        answer_text=None,
        video_fps=24.0,
    )

    metadata = captured["videos_kwargs"]["video_metadata"]
    assert len(metadata) == 2
    assert all(len(sample) == 2 for sample in metadata)
    assert all(
        isinstance(item, VideoMetadata) for sample in metadata for item in sample
    )
    assert metadata[0][0].total_num_frames == 5
    assert metadata[0][0].frames_indices == [0, 1, 2, 3, 4]
    assert metadata[1][1].total_num_frames == 5


# --------------------------------------------------------------------------
# VLM SFT collate: transformers >= 5 emits mm_token_type_ids
# --------------------------------------------------------------------------


def _vlm_sft_sample(idx, seq_len, mm_token_type_ids):
    """A sample shaped like ``Robo2VLMSFTDataset.encode_prompt`` output."""
    return SimpleNamespace(
        idx=idx,
        length=seq_len,
        prompt=torch.arange(1, seq_len + 1, dtype=torch.long),
        answer="a",
        solution="s",
        image_data=None,
        prompt_text="p",
        meta={},
        attention_mask=torch.ones(seq_len, dtype=torch.long),
        label_mask=torch.zeros(seq_len, dtype=torch.bool),
        multi_modal_inputs={
            "pixel_values": torch.randn(4, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "mm_token_type_ids": mm_token_type_ids,
        },
    )


def test_vlm_collate_left_pads_mm_token_type_ids_like_prompts():
    from rlinf.data.datasets.vlm.collate_fn import collate_fn

    # The three processor output shapes seen in the wild: (1, L) tensor,
    # plain list, and (L,) tensor.
    samples = [
        _vlm_sft_sample(0, 9, torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1, 0]])),
        _vlm_sft_sample(1, 5, [0, 1, 1, 0, 0]),
        _vlm_sft_sample(2, 7, torch.tensor([0, 1, 1, 1, 1, 0, 0])),
    ]

    batch = collate_fn(samples)

    mm_ids = batch["multi_modal_inputs"]["mm_token_type_ids"]
    assert mm_ids.shape == (3, 9)
    assert mm_ids.dtype == torch.long

    for i, sample in enumerate(samples):
        pad_len = 9 - sample.length
        # Left padding is 0 (text token) and masked by attention_mask.
        assert (mm_ids[i, :pad_len] == 0).all()
        assert (batch["attention_mask"][i, :pad_len] == 0).all()
        # Valid part matches the processor output bit-for-bit.
        expected = torch.as_tensor(
            sample.multi_modal_inputs["mm_token_type_ids"], dtype=torch.long
        ).flatten()
        assert torch.equal(mm_ids[i, pad_len:], expected)
        # Padding positions line up with the prompt's padding positions.
        assert (batch["prompt"][i, :pad_len] == 0).all()

    assert isinstance(batch["multi_modal_inputs"]["pixel_values"], list)
    assert batch["multi_modal_inputs"]["image_grid_thw"].shape == (3, 3)


def test_vlm_collate_stacks_mm_token_type_ids_without_padding():
    from rlinf.data.datasets.vlm.collate_fn import collate_fn

    samples = [
        _vlm_sft_sample(0, 5, torch.tensor([0, 1, 1, 0, 0])),
        _vlm_sft_sample(1, 5, torch.tensor([[0, 0, 1, 1, 0]])),
    ]

    batch = collate_fn(samples)

    mm_ids = batch["multi_modal_inputs"]["mm_token_type_ids"]
    assert mm_ids.shape == (2, 5)
    assert torch.equal(mm_ids[0], torch.tensor([0, 1, 1, 0, 0]))
    assert torch.equal(mm_ids[1], torch.tensor([0, 0, 1, 1, 0]))


def test_vlm_collate_still_rejects_unknown_mm_keys():
    from rlinf.data.datasets.vlm.collate_fn import collate_fn

    sample = _vlm_sft_sample(0, 5, torch.tensor([0, 1, 1, 0, 0]))
    sample.multi_modal_inputs["some_future_field"] = torch.zeros(5)

    with pytest.raises(ValueError, match="Unsupported multi_modal_input key"):
        collate_fn([sample])


@pytest.fixture
def fr3_recorder_config():
    from toolkits.realworld_check.validate_fr3_config import load_config

    return load_config(["runner.num_data_episodes=2", "env.eval.max_episode_steps=3"])


@pytest.fixture
def fr3_dataset(tmp_path, fr3_recorder_config):
    from rlinf.data.collectors.fr3_offline import collect_synthetic_dataset

    directory = tmp_path / "synthetic"
    report = collect_synthetic_dataset(fr3_recorder_config, directory)
    return directory, report


def test_fr3_native_recorder_round_trip(fr3_dataset):
    from rlinf.data.storage.fr3_synthetic import read_json, validate_synthetic_dataset

    root, report = fr3_dataset
    assert report["episodes"] == 2 and report["steps"] == 6
    assert report["native_reload"] and not report["physical_execution"]
    assert validate_synthetic_dataset(root) == report
    manifest = read_json(root / "manifest.json")
    assert manifest["status"] == "COMPLETE"
    assert manifest["camera_roles"] == {"d405": None, "d435i": None}
    assert manifest["physical_command_authorization"] == "NOT_GRANTED"
    assert not (root / ".buffer").exists()
    assert len(list((root / "demos").glob("trajectory_*.pt"))) == 2
    episodes = [read_json(root / e["metadata_path"]) for e in manifest["episodes"]]
    assert all(
        e["status"] == "COMPLETE" and e["termination_reason"] == "synthetic_horizon"
        for e in episodes
    )
    assert (
        episodes[0]["environment_episode_id"] != episodes[1]["environment_episode_id"]
    )
    assert episodes[0]["sessions"]["arm"] != episodes[1]["sessions"]["arm"]
    for file in (root / "demos").glob("trajectory_*.pt"):
        trajectory = torch.load(file, weights_only=True)
        assert trajectory["actions"].shape == (3, 1, 7)
        assert trajectory["curr_obs"]["states"].shape == (3, 1, 42)
        assert trajectory["curr_obs"]["main_images"].shape == (3, 1, 24, 32, 3)
        assert trajectory["curr_obs"]["extra_view_images"].shape == (3, 1, 1, 24, 32, 3)
        assert trajectory["curr_obs"]["main_images"].dtype == torch.uint8
        assert not trajectory["terminations"].any()
        assert trajectory["truncations"][-1].all()
        assert not trajectory["intervene_flags"].any()
        assert torch.count_nonzero(trajectory["rewards"]) == 0
        assert not torch.equal(
            trajectory["curr_obs"]["states"][0], trajectory["next_obs"]["states"][0]
        )
    records = [
        json.loads(line)
        for line in (root / manifest["episodes"][0]["provenance_path"])
        .read_text()
        .splitlines()
    ]
    requested, outcome = records[:2]
    assert requested["operator_request"]["physical_mapping_verified"] is False
    assert len(requested["operator_request"]["raw_input"]["axes"]) == 6
    assert (
        requested["operator_request"]["input_timing"]["clock_domain"]
        == manifest["synthetic_clock_domain"]
    )
    assert outcome["physical_grasp_success"] is None
    assert outcome["physical_controller_acceptance"] == "UNAVAILABLE"
    arm = requested["typed_requests"]["arm"]
    gripper = requested["typed_requests"]["end_effector"]
    assert arm["kind"] == "fr3_normalized_pose_offset_v1"
    assert gripper["kind"] == "robotiq_normalized_closure_v1"
    assert len(gripper["values"]) == 1 and 0 <= gripper["values"][0] <= 1
    assert arm["clock_domain"] == requested["recorder_clock_domain"]
    assert arm["clock_domain"] != requested["synthetic_clock_domain"]
    arm_states = [e["state"] for e in outcome["action_events"]["arm"]]
    assert arm_states[:5] == [
        "created",
        "validated",
        "admitted",
        "submitted",
        "acceptance_unknown",
    ]
    assert "mock_canceled" in arm_states
    assert "mock_completed" in [
        e["state"] for e in outcome["action_events"]["end_effector"]
    ]
    assert not any(str(root) in p.read_text() for p in root.rglob("*.json"))
    assert not any(str(root) in p.read_text() for p in root.rglob("*.jsonl"))


def test_fr3_recorder_reproducible_payloads(tmp_path, fr3_recorder_config):
    from rlinf.data.collectors.fr3_offline import collect_synthetic_dataset

    for name in ("first", "second"):
        collect_synthetic_dataset(fr3_recorder_config, tmp_path / name)
    for first in sorted((tmp_path / "first/demos").glob("trajectory_*.pt")):
        a = torch.load(first, weights_only=True)
        b = torch.load(tmp_path / "second/demos" / first.name, weights_only=True)
        for key in ("actions", "rewards", "dones", "intervene_flags"):
            assert torch.equal(a[key], b[key])
        for section in ("curr_obs", "next_obs"):
            assert all(
                torch.equal(value, b[section][key]) for key, value in a[section].items()
            )


@pytest.mark.parametrize(
    "scenario",
    [
        "invalid_action",
        "expired_action",
        "no_authorization",
        "mock_fail",
        "mock_reject",
        "ack_timeout",
        "worker_crash",
        "ipc_failure",
        "missing_camera",
        "delayed_camera",
        "stale_observation",
        "sample_skew",
        "interrupted",
    ],
)
def test_fr3_recorder_failures_are_incomplete(tmp_path, fr3_recorder_config, scenario):
    from rlinf.data.collectors.fr3_offline import (
        SyntheticScenario,
        collect_synthetic_dataset,
    )
    from rlinf.data.storage.fr3_synthetic import read_json, validate_synthetic_dataset

    if scenario == "ack_timeout":
        fr3_recorder_config.env.eval.override_cfg.synthetic_profile.acknowledgement_timeout_ns = 1
    root = tmp_path / "failed"
    exception = (
        KeyboardInterrupt
        if scenario == "interrupted"
        else (ValueError, ConnectionError)
    )
    with pytest.raises(exception):
        collect_synthetic_dataset(
            fr3_recorder_config, root, scenario=SyntheticScenario(scenario)
        )
    manifest = read_json(root / "manifest.json")
    assert manifest["status"] == "INCOMPLETE" and manifest["errors"]
    assert manifest["physical_execution"] is False
    episode = read_json(root / "episodes/episode_000000/episode.json")
    assert episode["status"] == "INCOMPLETE"
    records = [
        json.loads(line)
        for line in (root / "episodes/episode_000000/steps.jsonl")
        .read_text()
        .splitlines()
    ]
    assert records[-1]["event"] == "step_outcome" and records[-1]["valid"] is False
    if scenario == "ack_timeout":
        assert any(
            e["reason"] == "mock_ack_timeout"
            for e in records[-1]["action_events"]["end_effector"]
        )
    with pytest.raises(ValueError, match="not finalized"):
        validate_synthetic_dataset(root)


def test_fr3_recorder_rejects_native_stale_observation(tmp_path, fr3_recorder_config):
    from rlinf.data.collectors.fr3_offline import collect_synthetic_dataset

    fr3_recorder_config.env.eval.override_cfg.synthetic_profile.max_observation_age_ns = 1
    with pytest.raises(ValueError, match="observation"):
        collect_synthetic_dataset(fr3_recorder_config, tmp_path / "stale")
    assert (
        json.loads((tmp_path / "stale/manifest.json").read_text())["status"]
        == "INCOMPLETE"
    )


@pytest.mark.parametrize(
    "setting",
    [
        "env.eval.env_type=real",
        "env.eval.override_cfg.execution_mode=live",
        "env.eval.override_cfg.action_transport=ros2",
        "env.eval.data_collection.enabled=true",
        "runner.num_data_episodes=0",
        "offline_collection.schema_version=2",
    ],
)
def test_fr3_recorder_config_fails_before_resources(tmp_path, setting):
    from rlinf.data.collectors.fr3_offline import collect_synthetic_dataset
    from toolkits.realworld_check.validate_fr3_config import load_config

    root = tmp_path / "invalid"
    with pytest.raises(ValueError):
        collect_synthetic_dataset(load_config([setting]), root)
    assert not root.exists()


def test_fr3_recorder_never_overwrites(fr3_dataset, fr3_recorder_config):
    from rlinf.data.collectors.fr3_offline import collect_synthetic_dataset

    root, _ = fr3_dataset
    original = (root / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        collect_synthetic_dataset(fr3_recorder_config, root)
    assert (root / "manifest.json").read_bytes() == original


@pytest.mark.parametrize(
    "error",
    [OSError("synthetic disk full"), TypeError("synthetic serialization failure")],
)
def test_fr3_native_save_failure_cannot_finalize(
    tmp_path, fr3_recorder_config, monkeypatch, error
):
    from rlinf.data.collectors.fr3_offline import collect_synthetic_dataset
    from rlinf.data.storage.fr3_synthetic import validate_synthetic_dataset

    def fail_save(*args, **kwargs):
        raise error

    monkeypatch.setattr(torch, "save", fail_save)
    root = tmp_path / "failed-write"
    with pytest.raises(type(error)):
        collect_synthetic_dataset(fr3_recorder_config, root)
    assert json.loads((root / "manifest.json").read_text())["status"] == "INCOMPLETE"
    with pytest.raises(ValueError):
        validate_synthetic_dataset(root)


def test_fr3_status_write_failure_is_not_finalized(
    tmp_path, fr3_recorder_config, monkeypatch
):
    import os

    from rlinf.data.collectors.fr3_offline import collect_synthetic_dataset

    original = os.replace

    def fail_episode(source, target):
        if str(target).endswith("episode.json"):
            raise OSError("synthetic status write failure")
        return original(source, target)

    monkeypatch.setattr(os, "replace", fail_episode)
    root = tmp_path / "failed-status"
    with pytest.raises(OSError):
        collect_synthetic_dataset(fr3_recorder_config, root)
    assert json.loads((root / "manifest.json").read_text())["status"] == "INCOMPLETE"


def test_fr3_cleanup_failure_blocks_complete_marker(
    tmp_path, fr3_recorder_config, monkeypatch
):
    import subprocess

    from rlinf.data.collectors.fr3_offline import collect_synthetic_dataset

    fr3_recorder_config.runner.num_data_episodes = 1
    original = subprocess.Popen.wait
    affected = []

    def fail_first_wait(process, *args, **kwargs):
        result = original(process, *args, **kwargs)
        if not affected:
            affected.append(process)
            raise OSError("synthetic reaping failure")
        return result

    monkeypatch.setattr(subprocess.Popen, "wait", fail_first_wait)
    root = tmp_path / "failed-cleanup"
    with pytest.raises(Exception, match="synthetic reaping failure"):
        collect_synthetic_dataset(fr3_recorder_config, root)
    assert affected and all(p.poll() is not None for p in affected)
    assert json.loads((root / "manifest.json").read_text())["status"] == "INCOMPLETE"


@pytest.mark.parametrize(
    "corruption",
    [
        "schema",
        "physical_claim",
        "truncated_log",
        "file_bytes",
        "missing_file",
        "action_values",
        "frame_time",
        "logical_time",
        "operator_input",
        "dense_state",
    ],
)
def test_fr3_reload_detects_schema_and_corruption(fr3_dataset, corruption):
    from rlinf.data.storage.fr3_synthetic import file_digest, validate_synthetic_dataset

    root, _ = fr3_dataset
    manifest = json.loads((root / "manifest.json").read_text())
    if corruption == "schema":
        manifest["schema"] = "unknown"
    elif corruption == "physical_claim":
        manifest["physical_execution"] = True
    else:
        provenance = root / manifest["episodes"][0]["provenance_path"]
        if corruption == "truncated_log":
            provenance.write_bytes(provenance.read_bytes()[:-12])
        elif corruption in (
            "action_values",
            "frame_time",
            "logical_time",
            "operator_input",
        ):
            lines = [json.loads(line) for line in provenance.read_text().splitlines()]
            if corruption == "action_values":
                lines[0]["typed_requests"]["arm"]["values"][0] += 0.1
            elif corruption == "frame_time":
                lines[0]["observation_before"]["cameras"]["d405"]["metadata"][
                    "fixture_timing"
                ]["clock_domain"] = "other-host"
            elif corruption == "logical_time":
                lines[0]["logical_action_ns"] += 1
            else:
                lines[0]["operator_request"]["raw_input"]["axes"][0] += 0.1
            provenance.write_text("".join(json.dumps(line) + "\n" for line in lines))
            manifest["files"][str(provenance.relative_to(root))] = file_digest(
                provenance
            )
        else:
            native = next((root / "demos").glob("trajectory_*.pt"))
            if corruption == "missing_file":
                native.unlink()
            elif corruption == "dense_state":
                payload = torch.load(native, weights_only=True)
                payload["curr_obs"]["states"][0, 0, 0] = float("nan")
                torch.save(payload, native)
                manifest["files"][str(native.relative_to(root))] = file_digest(native)
            else:
                native.write_bytes(native.read_bytes()[:-10] + b"corruption")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_synthetic_dataset(root)


def test_fr3_async_alignment_and_no_substitution():
    import dataclasses

    from rlinf.robotics.parts.fr3_fixtures import (
        STREAMS,
        SyntheticFrame,
        validate_alignment,
    )

    frame = SyntheticFrame(0, 1_000_000_000, "synthetic:test")
    times = {s: frame.timing(s) for s in STREAMS}
    assert len({v["arrival_ns"] for v in times.values()}) > 1
    validate_alignment(times, 50_000_000, 20_000_000)
    for broken in (
        dataclasses.replace(frame, missing=("d435i",)),
        dataclasses.replace(
            frame, arrival_offsets_ns={**frame.arrival_offsets_ns, "d405": 1}
        ),
    ):
        with pytest.raises(ValueError):
            validate_alignment(
                {s: broken.timing(s) for s in STREAMS}, 50_000_000, 20_000_000
            )


def test_fr3_recorder_subprocess_is_hardware_inaccessible(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    program = r"""
import builtins
import numpy.testing
import os
import subprocess
import sys
import psutil
import ray
original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split('.')[0] in {'rclpy','rospy','franky','pyrealsense2','hid','hidapi','spnav','pyspacemouse','pymodbus'}:
        raise AssertionError('hardware SDK import: '+name)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import
psutil.process_iter = lambda *a, **k: (_ for _ in ()).throw(AssertionError('hardware/process discovery'))
ray.init = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Ray initialization'))
original_popen = subprocess.Popen
workers = []
class GuardedPopen(original_popen):
    def __init__(self, args, *pos, **kwargs):
        assert any(str(a).endswith('/mock_action_worker.py') for a in args), args
        assert not any(k.startswith(('ROS_', 'RMW_', 'FASTRTPS_', 'FASTDDS_', 'CYCLONEDDS_')) for k in kwargs['env'])
        super().__init__(args, *pos, **kwargs)
        workers.append(self)
subprocess.Popen = GuardedPopen
from toolkits.realworld_check.validate_fr3_config import load_config
from rlinf.data.collectors.fr3_offline import collect_synthetic_dataset
from rlinf.data.storage.fr3_synthetic import validate_synthetic_dataset
cfg=load_config(['runner.num_data_episodes=2','env.eval.max_episode_steps=2'])
report=collect_synthetic_dataset(cfg, sys.argv[1])
assert validate_synthetic_dataset(sys.argv[1]) == report
assert report['episodes']==2 and report['steps']==4 and not report['physical_execution']
assert 'rlinf.envs.real.franka' not in sys.modules
assert all(worker.poll() is not None for worker in workers)
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "guarded")],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "ROS_DOMAIN_ID": "70"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_fr3_offline_collection_cli(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = tmp_path / "cli"
    command = [
        sys.executable,
        "-m",
        "examples.embodiment.collect_fr3_offline",
        "--output",
        str(root),
    ]
    result = subprocess.run(
        command + ["runner.num_data_episodes=2", "env.eval.max_episode_steps=2"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["episodes"] == 2 and report["steps"] == 4
    result = subprocess.run(
        command + ["--validate-only"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == report


def test_fr3_offline_cli_live_override_opens_no_dataset(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = tmp_path / "forbidden"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.embodiment.collect_fr3_offline",
            "--output",
            str(root),
            "env.eval.env_type=real",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert not root.exists()


@pytest.mark.parametrize(
    ("missing", "reason"),
    [("arm", "complete synthetic"), ("d405", "missing_or_malformed_camera")],
)
def test_fr3_missing_initial_observation_rolls_back_owned_resources(
    fr3_recorder_config, missing, reason
):
    from rlinf.envs import get_env_cls
    from rlinf.robotics.parts.fr3_fixtures import SyntheticFrame

    env = get_env_cls("fr3_offline")(fr3_recorder_config.env.eval)
    try:
        env.offline_task.set_synthetic_frame(
            SyntheticFrame(0, 1_000_000_000, "synthetic:missing", missing=(missing,))
        )
        with pytest.raises(ValueError, match=reason):
            env.reset()
        assert not env.offline_task.robot.is_connected
        assert not any(
            part.is_connected for part in env.offline_task.robot.children.values()
        )
    finally:
        env.close()


def _fr3_dataset_hashes(root):
    from rlinf.data.storage.fr3_synthetic import file_digest

    return {
        str(p.relative_to(root)): file_digest(p) for p in root.rglob("*") if p.is_file()
    }


def _fr3_rehash(root):
    from rlinf.data.storage.fr3_synthetic import file_digest

    manifest = json.loads((root / "manifest.json").read_text())
    for relative in manifest["files"]:
        path = root / relative
        if path.is_file():
            manifest["files"][relative] = file_digest(path)
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_fr3_quality_report_and_non_destructive_reload(fr3_dataset):
    from rlinf.data.storage.fr3_validation import assess_synthetic_dataset

    root, _ = fr3_dataset
    original = _fr3_dataset_hashes(root)
    report = assess_synthetic_dataset(root)
    assert report["valid"], report["diagnostics"]
    assert report == assess_synthetic_dataset(root)
    assert original == _fr3_dataset_hashes(root)
    assert report["counts"]["valid_episodes"] == 2
    assert report["counts"]["valid_transitions"] == 6
    assert report["counts"]["invalid_transitions"] == 0
    skew = report["timestamp_statistics"]["cross_stream_skew_ns"]
    assert skew["count"] == 8  # Two initial snapshots plus six next snapshots.
    assert skew["min"] == skew["max"] == 7_000_000
    assert (
        report["timestamp_statistics"]["source_minus_arm_ns/d405"]["min"] == -5_000_000
    )
    assert report["readiness"]["native_rlinf_serialization"] == "PASS"
    assert report["readiness"]["openpi_pi05_schema"] == "UNQUALIFIED"
    assert report["readiness"]["real_world_demonstration"] == "NOT_QUALIFIED"
    assert all(v["failures"] == 0 for v in report["modalities"].values())
    assert report["source_manifest_sha256"] == original["manifest.json"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "content", ["{", "[]", '{"schema":1,"schema":2}', '{"schema":NaN}']
)
def test_fr3_quality_invalid_manifests_are_diagnostic(tmp_path, content):
    from rlinf.data.storage.fr3_validation import assess_synthetic_dataset

    (tmp_path / "manifest.json").write_text(content)
    report = assess_synthetic_dataset(tmp_path)
    assert not report["valid"]
    assert report["diagnostics"][0]["path"] == "manifest.json"
    assert report["diagnostics"][0]["error_type"]
    assert report["readiness"]["native_rlinf_serialization"] == "FAIL"


@pytest.mark.parametrize(
    "failure,diagnostic",
    [
        ("missing_episode", "Missing artifact"),
        ("corrupt_pt", "Native transition payload unavailable"),
        ("state_dim", "state [T,1,42]"),
        ("nonfinite", "finite"),
        ("quaternion", "unit xyzw quaternion"),
        ("action_kind", "discriminator"),
        ("action_range", "channel limits"),
        ("provenance", "typed_requests"),
        ("physical_claim", "physical_execution=false"),
        ("frame_order", "frame"),
        ("duplicate", "transition identifier"),
        ("missing_camera", "camera"),
        ("skew", "skew"),
        ("clock", "clock"),
        ("stale", "stale"),
        ("duplicate_timestamp", "timestamp"),
        ("incomplete", "Incomplete episode"),
        ("schema_version", "schema version"),
        ("joint_order", "joint names/order"),
        ("closure", "closure outside"),
        ("encoding", "encoding"),
        ("mouse_dimension", "SpaceMouse axes"),
        ("rejected", "Rejected transition"),
        ("expiry", "Expired"),
        ("source_identity", "source"),
        ("episode_id", "Duplicate episode"),
        ("index", "index identity"),
        ("forward_action", "forward_inputs.action"),
        ("receipt_order", "receipt"),
        ("reward_dtype", "rewards"),
        ("partial_log", "Incomplete JSONL"),
    ],
)
def test_fr3_quality_failure_injection(fr3_dataset, failure, diagnostic):
    from rlinf.data.storage.fr3_validation import assess_synthetic_dataset

    root, _ = fr3_dataset
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["episodes"][0]
    ep_path = root / entry["metadata_path"]
    log_path = root / entry["provenance_path"]
    records = [json.loads(s) for s in log_path.read_text().splitlines()]
    obs = records[0]["observation_before"]
    native = next((root / "demos").glob("trajectory_0_*.pt"))
    if failure == "missing_episode":
        ep_path.unlink()
    elif failure == "corrupt_pt":
        native.write_bytes(b"broken trajectory")
    elif failure in {"state_dim", "nonfinite", "forward_action", "reward_dtype"}:
        payload = torch.load(native, weights_only=True)
        if failure == "state_dim":
            payload["curr_obs"]["states"] = payload["curr_obs"]["states"][..., :-1]
        elif failure == "nonfinite":
            payload["next_obs"]["states"][0, 0, 0] = float("nan")
        elif failure == "forward_action":
            payload["forward_inputs"]["action"][0, 0, 0] += 0.1
        else:
            payload["rewards"] = payload["rewards"].double()
        torch.save(payload, native)
    elif failure in {"incomplete", "episode_id"}:
        ep = json.loads(ep_path.read_text())
        if failure == "incomplete":
            ep["status"] = "INCOMPLETE"
        else:
            ep2 = root / manifest["episodes"][1]["metadata_path"]
            e = json.loads(ep2.read_text())
            e["episode_id"] = ep["episode_id"]
            ep2.write_text(json.dumps(e))
        ep_path.write_text(json.dumps(ep))
    elif failure == "schema_version":
        manifest["observation_schema_version"] = 99
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "index":
        p = root / "demos/trajectory_index.json"
        idx = json.loads(p.read_text())
        idx["trajectory_index"]["0"]["num_samples"] += 1
        p.write_text(json.dumps(idx))
    elif failure == "partial_log":
        log_path.write_bytes(log_path.read_bytes()[:-2])
    else:
        if failure == "quaternion":
            obs["arm"]["state"]["tcp_pose"][3:] = [0.0] * 4
            obs["arm"]["ros2"]["streams"]["pose"]["sample"]["data"]["tcp_pose"][3:] = [
                0.0
            ] * 4
        elif failure == "action_kind":
            records[0]["typed_requests"]["arm"]["kind"] = "twist"
        elif failure == "action_range":
            records[0]["typed_requests"]["arm"]["values"][0] = 2.0
            records[0]["operator_request"]["arm_values"][0] = 2.0
            for e in records[1]["action_events"]["arm"]:
                e["request"]["values"][0] = 2.0
        elif failure == "provenance":
            del records[0]["typed_requests"]
        elif failure == "physical_claim":
            records[1]["physical_execution"] = True
        elif failure in {
            "frame_order",
            "skew",
            "clock",
            "stale",
            "duplicate_timestamp",
        }:
            timing = obs["cameras"]["d405"]["metadata"]["fixture_timing"]
            if failure == "frame_order":
                timing["frame_index"] += 1
            elif failure == "clock":
                timing["clock_domain"] = "other-host-monotonic"
            elif failure == "stale":
                timing["source_ns"] -= 100_000_000
            elif failure == "duplicate_timestamp":
                after = records[1]["observation_after"]["cameras"]["d405"]["metadata"][
                    "fixture_timing"
                ]
                after["source_ns"] = timing["source_ns"]
            else:
                timing["source_ns"] -= 30_000_000
        elif failure == "duplicate":
            records[2:4] = copy.deepcopy(records[0:2])
        elif failure == "missing_camera":
            del obs["cameras"]["d405"]
        elif failure == "joint_order":
            obs["arm"]["ros2"]["streams"]["joints"]["sample"]["data"]["names"].reverse()
        elif failure == "closure":
            obs["gripper"]["state"]["normalized_closure"] = 2.0
            obs["gripper"]["ros2"]["streams"]["normalized"]["sample"]["data"][
                "position"
            ] = [2.0]
        elif failure == "encoding":
            obs["cameras"]["d405"]["metadata"]["encoding"] = "BGR"
        elif failure == "mouse_dimension":
            obs["spacemouse"]["axes"].pop()
        elif failure == "rejected":
            records[1]["valid"] = False
        elif failure == "expiry":
            request = records[0]["typed_requests"]["arm"]
            request["deadline_ns"] = request["generated_ns"]
            for e in records[1]["action_events"]["arm"]:
                e["request"]["deadline_ns"] = request["deadline_ns"]
        elif failure == "source_identity":
            records[0]["operator_request"]["source_id"] = "physical-operator"
        elif failure == "receipt_order":
            obs["cameras"]["d405"]["metadata"]["receipt_monotonic_ns"] = (
                records[0]["recorder_ns"] + 1
            )
        log_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    # Update hashes so semantic checks, rather than only hash checks, are exercised.
    _fr3_rehash(root)
    original = _fr3_dataset_hashes(root)
    report = assess_synthetic_dataset(root)
    assert not report["valid"]
    messages = "\n".join(e["message"] for e in report["diagnostics"])
    assert diagnostic.lower() in messages.lower(), messages
    assert report["readiness"]["native_rlinf_serialization"] == "FAIL"
    assert original == _fr3_dataset_hashes(root)
    if failure == "incomplete":
        assert report["counts"]["incomplete_episodes"] == 1
    if failure in {"missing_episode", "corrupt_pt", "partial_log"}:
        assert report["counts"]["corrupted_episodes"] == 1
    if failure == "rejected":
        assert report["counts"]["rejected_transitions"] == 1


def test_fr3_quality_openpi_required_fields_remain_unqualified(fr3_dataset):
    from rlinf.data.storage.fr3_validation import assess_synthetic_dataset

    report = assess_synthetic_dataset(fr3_dataset[0])
    gates = report["downstream"]["openpi_pi05"]
    assert gates["status"] == "UNQUALIFIED"
    assert "task_instruction_missing_from_native_tensors" in gates["gaps"]
    assert "state_42_incompatible_with_realworld_input_19" in gates["gaps"]
    assert report["missing_features"]["task_instruction"] == 2
    assert "policy_checkpoint_version" in report["downstream"]["recap"]["missing"]


def test_fr3_lerobot_real_export_round_trip(fr3_dataset, tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    import datasets.config

    from rlinf.data.storage.lerobot.fr3_export import (
        audit_synthetic_export,
        export_synthetic_dataset,
    )

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    root, _ = fr3_dataset
    original = _fr3_dataset_hashes(root)
    dest = tmp_path / "export"
    report = export_synthetic_dataset(root, dest)
    assert report["status"] == "COMPLETE"
    audit = audit_synthetic_export(dest)
    assert audit == report["audit"]
    assert audit["episodes"] == 2 and audit["transitions"] == 6
    assert all(n == 6 for n in audit["fields_compared"].values())
    assert _fr3_dataset_hashes(dest / "native") == original
    with pytest.raises(FileExistsError):
        export_synthetic_dataset(root, dest)
    assert _fr3_dataset_hashes(root) == original
    # A rehashed frame-table mutation must fail field comparison on independent reload.
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = next((dest / "lerobot/data").rglob("*.parquet"))
    table = pq.read_table(parquet)
    column = table.column_names.index("actions")
    rows = table.column(column).to_pylist()
    rows[0][0] += 0.1
    table = table.set_column(
        column, table.field(column), pa.array(rows, type=table.field(column).type)
    )
    pq.write_table(table, parquet)
    from rlinf.data.storage.fr3_synthetic import file_digest

    metadata = json.loads((dest / "export.json").read_text())
    metadata["files"][str(parquet.relative_to(dest))] = file_digest(parquet)
    (dest / "export.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="field differs.*actions"):
        audit_synthetic_export(dest)


@pytest.mark.parametrize("label", ["", "pick up object", "unknown task", None])
def test_fr3_lerobot_invalid_task_labels_create_no_output(fr3_dataset, tmp_path, label):
    from rlinf.data.storage.lerobot.fr3_export import export_synthetic_dataset

    dest = tmp_path / "invalid-export"
    with pytest.raises(ValueError, match="synthetic task label"):
        export_synthetic_dataset(fr3_dataset[0], dest, task_label=label)
    assert not dest.exists()


@pytest.mark.parametrize("stage", ["create", "reload"])
def test_fr3_lerobot_failure_preserves_audit(fr3_dataset, tmp_path, monkeypatch, stage):
    pytest.importorskip("lerobot")
    import datasets.config
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rlinf.data.storage.lerobot.fr3_export import export_synthetic_dataset

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")

    def fail(*args, **kwargs):
        raise RuntimeError("injected LeRobot " + stage)

    monkeypatch.setattr(
        LeRobotDataset, "create" if stage == "create" else "__init__", fail
    )
    original = _fr3_dataset_hashes(fr3_dataset[0])
    dest = tmp_path / "failed-export"
    with pytest.raises(RuntimeError, match="injected LeRobot"):
        export_synthetic_dataset(fr3_dataset[0], dest)
    report = json.loads((dest / "export.json").read_text())
    assert report["status"] == "FAILED"
    assert report["physical_execution"] is False
    assert report["error"]["type"] == "RuntimeError"
    assert _fr3_dataset_hashes(fr3_dataset[0]) == original


def test_fr3_quality_cli_hardware_inaccessible(fr3_dataset, tmp_path):
    import os
    import subprocess
    import sys

    program = r"""
import builtins, sys
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split('.')[0] in {'rclpy','rospy','franky','pyrealsense2','hid','hidapi','spnav','pyspacemouse','pymodbus'}:
        raise AssertionError('hardware SDK import: ' + name)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
from toolkits.realworld_check.validate_fr3_dataset import main
raise SystemExit(main())
"""
    source, output = fr3_dataset[0], tmp_path / "quality"
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_CACHE": str(tmp_path / "hf-cache"),
    }
    command = [
        sys.executable,
        "-c",
        program,
        "--dataset",
        str(source),
        "--output",
        str(output),
    ]
    first = subprocess.run(
        [*command, "--export"], env=env, capture_output=True, text=True, timeout=60
    )
    assert first.returncode == 0, first.stdout + first.stderr
    reload = subprocess.run(
        [
            *command[:-1],
            str(tmp_path / "reload"),
            "--reload-export",
            str(output / "export"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert reload.returncode == 0, reload.stdout + reload.stderr
    report = json.loads((tmp_path / "reload/quality.json").read_text())
    assert report["export"]["audit"]["transitions"] == 6
    original = _fr3_dataset_hashes(source)
    fail = subprocess.run(
        command[:-1] + [str(source / "quality")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert fail.returncode == 1
    assert _fr3_dataset_hashes(source) == original


def test_fr3_quality_retains_unindexed_incomplete_episode(
    tmp_path, fr3_recorder_config
):
    from rlinf.data.collectors.fr3_offline import (
        SyntheticScenario,
        collect_synthetic_dataset,
    )
    from rlinf.data.storage.fr3_validation import assess_synthetic_dataset

    root = tmp_path / "incomplete"
    with pytest.raises(ValueError):
        collect_synthetic_dataset(
            fr3_recorder_config, root, scenario=SyntheticScenario("invalid_action")
        )
    original = _fr3_dataset_hashes(root)
    report = assess_synthetic_dataset(root)
    assert not report["valid"]
    assert report["counts"]["incomplete_episodes"] == 1
    assert report["counts"]["rejected_transitions"] == 1
    assert report["unindexed_artifacts"]
    assert original == _fr3_dataset_hashes(root)


@pytest.mark.parametrize("events", [[], {"arm": None}, {"arm": [None]}])
def test_fr3_quality_malformed_action_journal_is_diagnostic(fr3_dataset, events):
    from rlinf.data.storage.fr3_validation import assess_synthetic_dataset

    root, _ = fr3_dataset
    path = root / "episodes/episode_000000/steps.jsonl"
    records = [json.loads(s) for s in path.read_text().splitlines()]
    records[1]["action_events"] = events
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    _fr3_rehash(root)
    report = assess_synthetic_dataset(root)
    assert not report["valid"]
    assert any(
        e["category"] == "actions" and e["path"] == "episodes/0/steps/0"
        for e in report["diagnostics"]
    )


@pytest.mark.parametrize(
    "relative", ["../outside.json", "/tmp/outside.json", "symlink.json"]
)
def test_fr3_quality_unsafe_artifacts_are_diagnostic(fr3_dataset, tmp_path, relative):
    from rlinf.data.storage.fr3_synthetic import file_digest
    from rlinf.data.storage.fr3_validation import assess_synthetic_dataset

    root, _ = fr3_dataset
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    if relative == "symlink.json":
        (root / relative).symlink_to(outside)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][relative] = file_digest(outside)
    path.write_text(json.dumps(manifest))
    report = assess_synthetic_dataset(root)
    assert not report["valid"]
    assert any(
        "artifact path" in e["message"] or "Symlink artifact" in e["message"]
        for e in report["diagnostics"]
    )
    assert outside.read_text() == "{}"


@pytest.mark.parametrize(
    "field", ["step_period_ns", "episode_path", "metadata_path", "episode_start"]
)
def test_fr3_quality_malformed_nested_schema_is_diagnostic(fr3_dataset, field):
    from rlinf.data.storage.fr3_validation import assess_synthetic_dataset

    root, _ = fr3_dataset
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    if field == "step_period_ns":
        manifest[field] = None
    elif field == "episode_path":
        manifest["episodes"][0]["provenance_path"] = []
    elif field == "metadata_path":
        manifest["episodes"][0]["metadata_path"] = []
    else:
        episode_path = root / manifest["episodes"][0]["metadata_path"]
        episode = json.loads(episode_path.read_text())
        episode["start_logical_ns"] = None
        episode_path.write_text(json.dumps(episode))
    path.write_text(json.dumps(manifest))
    _fr3_rehash(root)
    report = assess_synthetic_dataset(root)
    assert not report["valid"]
    assert report["diagnostics"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("physical_execution", True),
        ("last_disposition", "rejected"),
        ("session_id", "other-session"),
    ],
)
def test_fr3_quality_mock_snapshot_must_agree_with_journal(fr3_dataset, field, value):
    from rlinf.data.storage.fr3_validation import assess_synthetic_dataset

    root, _ = fr3_dataset
    path = root / "episodes/episode_000000/steps.jsonl"
    records = [json.loads(s) for s in path.read_text().splitlines()]
    records[1]["mock_states"]["arm"][field] = value
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    _fr3_rehash(root)
    report = assess_synthetic_dataset(root)
    assert not report["valid"]
    assert report["readiness"]["typed_action_provenance"] == "FAIL"


@pytest.fixture(scope="module")
def fr3_openpi_export(tmp_path_factory):
    pytest.importorskip(
        "openpi",
        reason="FR3 structural qualification requires the pinned optional OpenPI package",
    )
    import datasets.config

    from rlinf.data.collectors.fr3_offline import collect_synthetic_dataset
    from rlinf.data.storage.lerobot.fr3_export import export_synthetic_dataset
    from toolkits.realworld_check.validate_fr3_config import load_config

    root = tmp_path_factory.mktemp("fr3-openpi")
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(datasets.config, "HF_DATASETS_CACHE", root / "hf-cache")
        cfg = load_config(
            [
                "runner.num_data_episodes=2",
                "env.eval.max_episode_steps=4",
                "env.eval.seed=7",
            ]
        )
        collect_synthetic_dataset(cfg, root / "native")
        export_synthetic_dataset(root / "native", root / "export")
    return root / "export"


@pytest.fixture(scope="module")
def fr3_openpi_dataset(fr3_openpi_export):
    from rlinf.data.datasets.openpi.fr3 import FR3OfflineDataset

    return FR3OfflineDataset(fr3_openpi_export)


def _fr3_input_from_sample(sample):
    return {
        "state": sample["native_state"].copy(),
        "actions": sample["recorded_controller_requests"].copy(),
        "image": sample["raw_images"]["d435i"].copy(),
        "extra_view_image": sample["raw_images"]["d405"].copy(),
        "actions_is_pad": sample["actions_is_pad"].copy(),
        "fr3_context": copy.deepcopy(sample["provenance"]),
    }


def _fr3_norm_fixture(method="zscore"):
    from rlinf.data.datasets.openpi.fr3_contract import CONTRACT_ID

    stats = {
        key: {"mean": [0.25] * dim, "std": [2.0] * dim}
        for key, dim in (("state", 42), ("actions", 7))
    }
    if method == "quantile":
        for dim, entry in zip((42, 7), stats.values()):
            entry.update(q01=[-2.0] * dim, q99=[3.0] * dim)
    return {
        "schema": "fr3_normalization_fixture_v1",
        "scope": "synthetic_math_only",
        "contract_id": CONTRACT_ID,
        "method": method,
        "statistics": stats,
    }


def test_fr3_openpi_end_to_end_native_semantics(fr3_openpi_dataset):
    from openpi.models.model import Observation
    from openpi.transforms import ResizeImages

    ds = fr3_openpi_dataset
    original = _fr3_dataset_hashes(ds.root)
    assert len(ds) == 8
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["state"].shape == sample["native_state"].shape == (42,)
        assert np.array_equal(sample["state"], sample["native_state"])
        assert sample["recorded_controller_requests"].shape == (4, 7)
        assert (
            sample["model_training_target"]
            is sample["predicted_model_action"]
            is sample["prompt"]
            is None
        )
        assert sample["checkpoint_status"] == "CHECKPOINT_UNVERIFIED"
        assert sample["provenance"]["episode_index"] == i // 4
        assert sample["provenance"]["frame_index"] == i % 4
        assert sample["provenance"]["metadata"]["task"] is None
        assert (
            sample["provenance"]["operator_request"]["physical_mapping_verified"]
            is False
        )
        assert (
            sample["provenance"]["typed_requests"]["arm"]["kind"]
            == "fr3_normalized_pose_offset_v1"
        )
        assert sample["provenance"]["mock_outcome"]["physical_grasp_success"] is None
        assert sample["image_mask"] == {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
            "right_wrist_0_rgb": False,
        }
        for sensor, slot in (("d435i", "base_0_rgb"), ("d405", "left_wrist_0_rgb")):
            raw = sample["raw_images"][sensor]
            resized = np.asarray(
                ResizeImages(224, 224)({"image": {"camera": raw}})["image"]["camera"]
            )
            np.testing.assert_array_equal(
                sample["image"][slot], resized.astype(np.float32) / 255 * 2 - 1
            )
            assert sample["image"][slot].shape == (224, 224, 3)
        observation = Observation.from_dict(
            {
                "state": sample["state"],
                "image": sample["image"],
                "image_mask": sample["image_mask"],
            }
        )
        assert observation.state.shape == (42,)
        assert observation.tokenized_prompt is None
        assert not observation.image_masks["right_wrist_0_rgb"]
        sample["state"][0] = 123
        sample["provenance"]["metadata"]["split"] = "test"
        assert ds[i]["state"][0] != 123
        assert ds.metadata_document()["episodes"][i // 4]["split"] == "unassigned"
    assert _fr3_dataset_hashes(ds.root) == original
    assert ds.metadata_report["missing"]["task_instruction"] == 2
    with pytest.raises(ValueError, match="CHECKPOINT_UNVERIFIED"):
        ds.require_model_ready()


@pytest.mark.parametrize("horizon", [1, 2, 4, 20, 50])
def test_fr3_openpi_native_windows_never_cross_episodes(fr3_openpi_export, horizon):
    from rlinf.data.datasets.openpi.fr3 import FR3DatasetConfig, FR3OfflineDataset

    ds = FR3OfflineDataset(
        fr3_openpi_export, config=FR3DatasetConfig(action_horizon=horizon)
    )
    for i in range(len(ds)):
        sample = ds[i]
        step = i % 4
        indices = np.minimum(np.arange(step, step + horizon), 3)
        np.testing.assert_array_equal(
            sample["provenance"]["source_frame_indices"], indices
        )
        np.testing.assert_array_equal(
            sample["actions_is_pad"], np.arange(step, step + horizon) >= 4
        )
        assert sample["recorded_controller_requests"].shape == (horizon, 7)
        times = np.asarray(sample["provenance"]["source_timestamps_ns"])
        assert (np.diff(times[~sample["actions_is_pad"]]) == 100_000_000).all()
    last = ds[3]
    np.testing.assert_array_equal(
        last["recorded_controller_requests"],
        np.repeat(last["recorded_controller_requests"][:1], horizon, axis=0),
    )
    assert not np.array_equal(
        ds[3]["recorded_controller_requests"][0],
        ds[4]["recorded_controller_requests"][0],
    )


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5, 1001])
def test_fr3_openpi_invalid_horizon_is_rejected(horizon):
    from rlinf.data.datasets.openpi.fr3 import FR3DatasetConfig

    with pytest.raises(ValueError, match="horizon"):
        FR3DatasetConfig(action_horizon=horizon).validate()


def test_fr3_openpi_strict_short_windows_reject_missing_future(fr3_openpi_export):
    from rlinf.data.datasets.openpi.fr3 import FR3DatasetConfig, FR3OfflineDataset

    ds = FR3OfflineDataset(
        fr3_openpi_export,
        config=FR3DatasetConfig(
            action_horizon=4, padding_policy="require_full_window_v1"
        ),
    )
    assert not ds[0]["actions_is_pad"].any()
    with pytest.raises(ValueError, match="Missing future actions"):
        ds[3]


@pytest.mark.parametrize(
    "failure",
    [
        "ordering",
        "missing_feature",
        "dimension19",
        "quaternion",
        "joints",
        "action_kind",
        "absolute",
        "gripper",
        "frame",
        "camera_role",
        "physical",
        "missing_camera",
        "mask",
        "nonfinite",
        "permuted_values",
        "swapped_images",
    ],
)
def test_fr3_openpi_structural_input_failure_injection(fr3_openpi_dataset, failure):
    data = _fr3_input_from_sample(fr3_openpi_dataset[0])
    context = data["fr3_context"]
    if failure == "ordering":
        context["state_fields"].reverse()
    elif failure == "missing_feature":
        context["state_fields"].pop()
    elif failure == "dimension19":
        data["state"] = data["state"][:19]
    elif failure == "quaternion":
        data["state"][35:39] = 0
    elif failure == "joints":
        context["joint_names"].reverse()
    elif failure == "action_kind":
        context["action_kinds"][0] = "twist"
    elif failure == "absolute":
        context["action_semantics"] = "absolute_xyz_rpy"
    elif failure == "gripper":
        data["actions"][:, 6] = 85.0
    elif failure == "frame":
        context["frame_contract"]["tcp_pose"] = "calibrated_tcp"
    elif failure == "camera_role":
        context["camera_roles"]["d405"] = "wrist"
    elif failure == "physical":
        context["physical_execution"] = True
    elif failure == "missing_camera":
        data["extra_view_image"] = None
    elif failure == "mask":
        data["actions_is_pad"] = np.ones(4, dtype=np.float32)
    elif failure == "permuted_values":
        data["state"][:14] = np.concatenate([data["state"][7:14], data["state"][:7]])
    elif failure == "swapped_images":
        data["image"], data["extra_view_image"] = (
            data["extra_view_image"],
            data["image"],
        )
    else:
        data["state"][0] = float("nan")
    with pytest.raises(ValueError):
        fr3_openpi_dataset._transform(data)


@pytest.mark.parametrize("dimension", [19, 32])
def test_fr3_openpi_model_dimensions_cannot_be_silently_adapted(dimension):
    from rlinf.models.embodiment.openpi.policies.fr3_policy import FR3Inputs

    with pytest.raises(ValueError, match="42 state / 7 request"):
        FR3Inputs(state_dimension=dimension)


def test_fr3_openpi_transform_composes_without_enabling_prediction(fr3_openpi_dataset):
    from rlinf.data.datasets.openpi.transform_fn import compose
    from rlinf.models.embodiment.openpi.policies.fr3_policy import FR3Outputs

    result = compose([fr3_openpi_dataset._transform])(
        _fr3_input_from_sample(fr3_openpi_dataset[0])
    )
    assert result["state"].shape == (42,)
    with pytest.raises(ValueError, match="CHECKPOINT_UNVERIFIED"):
        FR3Outputs()({"actions": np.zeros((4, 7), dtype=np.float32)})


@pytest.mark.parametrize("method", ["zscore", "quantile"])
def test_fr3_openpi_synthetic_normalization_uses_native_math(
    method, fr3_openpi_dataset
):
    from rlinf.data.datasets.openpi.fr3_contract import normalization_fixture_check

    state = np.linspace(-1, 1, 42, dtype=np.float32)
    actions = np.linspace(-1, 1, 28, dtype=np.float32).reshape(4, 7)
    before = state.copy(), actions.copy()
    result = normalization_fixture_check(state, actions, _fr3_norm_fixture(method))
    expected = (
        (state - 0.25) / (2 + 1e-6)
        if method == "zscore"
        else (state + 2) / (5 + 1e-6) * 2 - 1
    )
    np.testing.assert_allclose(result["normalized"]["state"], expected, atol=1e-6)
    np.testing.assert_allclose(result["restored"]["actions"], actions, atol=1e-6)
    np.testing.assert_array_equal(state, before[0])
    np.testing.assert_array_equal(actions, before[1])
    assert result["checkpoint_qualification"] == "UNQUALIFIED"


@pytest.mark.parametrize(
    "failure",
    ["missing", "scope", "contract", "dimension", "zero_std", "nonfinite", "quantiles"],
)
def test_fr3_openpi_invalid_normalization_metadata(failure, fr3_openpi_dataset):
    from rlinf.data.datasets.openpi.fr3_contract import normalization_fixture_check

    fixture = _fr3_norm_fixture("quantile")
    if failure == "missing":
        fixture["statistics"] = None
    elif failure == "scope":
        fixture["scope"] = "qualified_checkpoint_assets"
    elif failure == "contract":
        fixture["contract_id"] = "absolute_pose"
    elif failure == "dimension":
        fixture["statistics"]["state"]["mean"].pop()
    elif failure == "zero_std":
        fixture["statistics"]["actions"]["std"][0] = 0
    elif failure == "nonfinite":
        fixture["statistics"]["actions"]["q01"][0] = float("nan")
    else:
        fixture["statistics"]["actions"]["q99"][0] = -3
    with pytest.raises(ValueError):
        normalization_fixture_check(
            np.zeros(42, dtype=np.float32), np.zeros((4, 7), dtype=np.float32), fixture
        )


def test_fr3_openpi_contract_tampering_fails_before_loading(tmp_path):
    from rlinf.data.datasets.openpi.fr3_contract import CONTRACT_PATH, load_fr3_contract

    data = json.loads(CONTRACT_PATH.read_text())
    data["checkpoint"]["execution_allowed"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fingerprint"):
        load_fr3_contract(path)


@pytest.mark.parametrize(
    "failure",
    [
        "leakage",
        "missing",
        "task",
        "source",
        "checkpoint",
        "physical",
        "manifest",
        "reward",
        "unknown_field",
    ],
)
def test_fr3_recap_metadata_failure_injection(fr3_openpi_dataset, tmp_path, failure):
    from rlinf.data.datasets.openpi.fr3 import FR3OfflineDataset

    doc = fr3_openpi_dataset.metadata_document()
    record = doc["episodes"][0]
    if failure == "leakage":
        other = copy.deepcopy(record)
        other["split"] = "test"
        record["split"] = "train"
        doc["episodes"].append(other)
    elif failure == "missing":
        doc["episodes"].pop()
    elif failure == "task":
        record["task"] = {
            "task_id": "task",
            "instruction": "",
            "version": "1",
            "origin": "recorded_original",
            "provenance": {},
        }
    elif failure == "source":
        record["source_type"] = "human_demonstration"
    elif failure == "checkpoint":
        record["policy"] = {
            "policy_id": "unknown",
            "checkpoint_sha256": "unknown",
            "configuration_sha256": "unknown",
        }
    elif failure == "physical":
        record["physical_execution"] = True
    elif failure == "manifest":
        record["source_manifest_sha256"] = "0" * 64
    elif failure == "reward":
        record["frame_annotations"] = [
            {"frame_index": 0, "reward": 1.0, "provenance": {}}
        ]
    else:
        record["silently_ignored"] = True
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        FR3OfflineDataset(fr3_openpi_dataset.root, annotations_path=path)


def test_fr3_recap_typed_annotations_are_synthetic_and_episode_bound(
    fr3_openpi_dataset, tmp_path
):
    from dataclasses import asdict

    from rlinf.data.datasets.openpi.fr3 import FR3OfflineDataset
    from rlinf.data.datasets.recap.fr3_metadata import (
        AnnotationProvenance,
        FrameAnnotation,
        Intervention,
        Outcome,
        TaskIdentity,
    )

    provenance = AnnotationProvenance(
        "synthetic:test",
        "literal_schema_fixture",
        "1",
        1,
        "synthetic:annotations",
        "synthetic_fixture",
    )
    doc = fr3_openpi_dataset.metadata_document()
    a, b = doc["episodes"]
    a["split"], b["split"] = "train", "test"
    a["task"] = asdict(
        TaskIdentity(
            "synthetic:task",
            "synthetic schema test only",
            "1",
            "synthetic_fixture",
            provenance,
        )
    )
    a["interventions"] = [
        asdict(Intervention(0, 1, "synthetic:operator", "schema test only", provenance))
    ]
    a["frame_annotations"] = [
        asdict(
            FrameAnnotation(
                0,
                provenance,
                reward=0.0,
                return_value=0.0,
                advantage_score=-1.0,
                advantage_label=False,
            )
        )
    ]
    a["outcome"] = asdict(Outcome("unscored", None, provenance))
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(doc))
    ds = FR3OfflineDataset(fr3_openpi_dataset.root, annotations_path=path)
    sample = ds[0]
    assert sample["provenance"]["metadata"]["split"] == "train"
    assert sample["provenance"]["metadata"]["frame_annotations"][0]["reward"] == 0
    assert sample["prompt"] is None
    assert ds[4]["provenance"]["metadata"]["split"] == "test"
    assert ds.metadata_report["experimental_readiness"] == "UNQUALIFIED_SYNTHETIC_ONLY"


def test_fr3_openpi_changed_timestamp_archive_is_rejected(fr3_openpi_export, tmp_path):
    from rlinf.data.datasets.openpi.fr3 import FR3OfflineDataset
    from rlinf.data.storage.fr3_synthetic import file_digest

    copied = tmp_path / "changed-export"
    shutil.copytree(fr3_openpi_export, copied)
    native = copied / "native"
    path = native / "episodes/episode_000000/steps.jsonl"
    records = [json.loads(s) for s in path.read_text().splitlines()]
    records[2]["logical_action_ns"] += 1
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    _fr3_rehash(native)
    manifest = json.loads((copied / "export.json").read_text())
    manifest["source_manifest_sha256"] = file_digest(native / "manifest.json")
    manifest["files"] = {
        str(p.relative_to(copied)): file_digest(p)
        for p in copied.rglob("*")
        if p.is_file() and p != copied / "export.json"
    }
    (copied / "export.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance archive"):
        FR3OfflineDataset(copied)


def test_fr3_openpi_cli_is_hardware_and_model_worker_inaccessible(
    fr3_openpi_export, tmp_path
):
    import os
    import subprocess
    import sys

    program = r"""
import builtins, subprocess, psutil, ray
original = builtins.__import__
original_popen = subprocess.Popen
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split('.')[0] in {'rclpy','rospy','franky','pyrealsense2','hid','hidapi','spnav','pyspacemouse','pymodbus'}:
        raise AssertionError('hardware SDK import: '+name)
    return original(name, globals, locals, fromlist, level)
def forbidden(*args, **kwargs):
    raise AssertionError('worker/process discovery or launch forbidden')
def checked_popen(args, *pos, **kwargs):
    if (args == 'lscpu' or args == ['uname', '-p']) and not kwargs.get('shell', False):
        return original_popen(args, *pos, **kwargs)
    return forbidden(args, *pos, **kwargs)
builtins.__import__=guarded
import numpy.testing
from rlinf.envs.real.fr3 import config as offline_schema
subprocess.Popen=checked_popen
psutil.process_iter=forbidden
ray.init=forbidden
from toolkits.realworld_check.qualify_fr3_openpi import main
raise SystemExit(main())
"""
    original = _fr3_dataset_hashes(fr3_openpi_export)
    dest = tmp_path / "qualification"
    command = [
        sys.executable,
        "-c",
        program,
        "--export",
        str(fr3_openpi_export),
        "--output",
        str(dest),
        "--horizon",
        "20",
        "--check-normalization-fixture",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_CACHE": str(tmp_path / "hf-cache"),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((dest / "qualification.json").read_text())
    assert report["structural_validation_complete"]
    assert report["transitions"] == 8 and report["windows"]["padded_windows"] == 8
    assert report["decisions"]["checkpoint_compatibility"] == "CHECKPOINT_UNVERIFIED"
    assert report["normalization_fixture"]["scope"] == "synthetic_math_only"
    assert report["physical_command_authorization"] == "NOT_GRANTED"
    assert original == _fr3_dataset_hashes(fr3_openpi_export)


def test_fr3_openpi_unknown_package_version_fails(monkeypatch):
    import importlib.metadata

    from rlinf.data.datasets.openpi.fr3_contract import (
        load_fr3_contract,
        verify_openpi_installation,
    )

    original = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "unknown" if name == "rlinf-openpi" else original(name),
    )
    with pytest.raises(ValueError, match="distribution version"):
        verify_openpi_installation(load_fr3_contract())


def test_fr3_openpi_modified_source_fails_pin(
    fr3_openpi_dataset, monkeypatch, tmp_path
):
    import importlib.util
    from pathlib import Path

    from rlinf.data.datasets.openpi.fr3_contract import (
        load_fr3_contract,
        verify_openpi_installation,
    )

    contract = load_fr3_contract()
    original = importlib.util.find_spec
    source = Path(original("openpi").origin).parent
    root = tmp_path / "openpi"
    for relative in contract["openpi"]["python_files_sha256"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    with (root / "transforms.py").open("a") as stream:
        stream.write("\n# modified external source fixture\n")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(origin=str(root / "__init__.py"))
            if name == "openpi"
            else original(name)
        ),
    )
    with pytest.raises(ValueError, match="qualified source pin"):
        verify_openpi_installation(contract)


def test_fr3_recap_original_instruction_cannot_be_invented(
    fr3_openpi_dataset, tmp_path
):
    from dataclasses import asdict

    from rlinf.data.datasets.openpi.fr3 import FR3OfflineDataset
    from rlinf.data.datasets.recap.fr3_metadata import (
        AnnotationProvenance,
        TaskIdentity,
    )

    provenance = AnnotationProvenance(
        "test:posthoc",
        "posthoc_claim",
        "1",
        1,
        "test:annotation-clock",
        "human_annotation",
    )
    doc = fr3_openpi_dataset.metadata_document()
    doc["episodes"][0]["task"] = asdict(
        TaskIdentity(
            "claimed_task",
            "not recorded in original dataset",
            "1",
            "recorded_original",
            provenance,
        )
    )
    path = tmp_path / "unverified-original-task.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Original task instruction is missing"):
        FR3OfflineDataset(fr3_openpi_dataset.root, annotations_path=path)


def test_fr3_recap_unknown_schema_fails(fr3_openpi_dataset, tmp_path):
    from rlinf.data.datasets.openpi.fr3 import FR3OfflineDataset

    doc = fr3_openpi_dataset.metadata_document()
    doc["schema"] = "unverified_recap_metadata_v2"
    path = tmp_path / "unknown-schema.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="metadata schema"):
        FR3OfflineDataset(fr3_openpi_dataset.root, annotations_path=path)


def test_fr3_recap_explicit_split_selection_has_no_leakage(
    fr3_openpi_dataset, tmp_path
):
    from rlinf.data.datasets.openpi.fr3 import FR3DatasetConfig, FR3OfflineDataset

    doc = fr3_openpi_dataset.metadata_document()
    doc["episodes"][0]["split"], doc["episodes"][1]["split"] = "train", "test"
    path = tmp_path / "split-annotations.json"
    path.write_text(json.dumps(doc))
    for split, episode_index in (("train", 0), ("test", 1)):
        ds = FR3OfflineDataset(
            fr3_openpi_dataset.root,
            config=FR3DatasetConfig(split=split),
            annotations_path=path,
        )
        assert len(ds) == 4
        assert all(
            ds[i]["provenance"]["episode_index"] == episode_index
            for i in range(len(ds))
        )
        assert all(
            ds[i]["provenance"]["metadata"]["split"] == split for i in range(len(ds))
        )
    with pytest.raises(ValueError, match="No episodes assigned"):
        FR3OfflineDataset(
            fr3_openpi_dataset.root,
            config=FR3DatasetConfig(split="validation"),
            annotations_path=path,
        )
    with pytest.raises(ValueError, match="No episodes assigned"):
        FR3OfflineDataset(
            fr3_openpi_dataset.root, config=FR3DatasetConfig(split="train")
        )


@pytest.fixture(scope="module")
def fr3_checkpoint_assets():
    import os
    from pathlib import Path

    raw = os.environ.get("FR3_PI05_ASSET_ROOT")
    if raw is None:
        pytest.skip(
            "Pinned small checkpoint assets require FR3_PI05_ASSET_ROOT; no test downloads assets"
        )
    return Path(raw).resolve()


def test_fr3_checkpoint_candidate_pins_complete_generation_set():
    import hashlib

    from rlinf.data.datasets.openpi.fr3_training import load_checkpoint_candidate

    candidate = load_checkpoint_candidate()
    objects = candidate["objects"]
    digest = hashlib.sha256(
        json.dumps(objects, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert candidate["candidate_id"] == "pi05_base_gcs_generation_set:" + digest
    assert len(objects) == len({o["name"] for o in objects}) == 29
    assert all(o["generation"].isdigit() and o["crc32c"] for o in objects)
    assert sum(o["size"] for o in objects) == 12_441_749_581
    assert candidate["configuration"]["pretraining_horizon"] is None
    assert candidate["training_proposal"]["execution_allowed"] is False


@pytest.mark.parametrize(
    "field", ["uri", "objects", "tokenizer", "architecture", "physical_execution"]
)
def test_fr3_checkpoint_changed_identity_or_assets_fail(tmp_path, field):
    from rlinf.data.datasets.openpi.fr3_training import (
        CANDIDATE_PATH,
        load_checkpoint_candidate,
    )

    candidate = json.loads(CANDIDATE_PATH.read_text())
    candidate[field] = "unreviewed"
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint_candidate(path)


def test_fr3_checkpoint_asset_shapes_do_not_qualify_fr3_statistics(
    fr3_checkpoint_assets,
):
    from rlinf.data.datasets.openpi.fr3_training import inspect_checkpoint_assets

    report = inspect_checkpoint_assets(fr3_checkpoint_assets)
    assert report["files_verified"] == 6
    assert report["downloaded_asset_bytes"] == 4_298_682
    assert report["normalization_dimensions"] == {
        "franka": {"state": 8, "actions": 8},
        "droid": {"state": 8, "actions": 8},
    }
    assert report["fr3_normalization_compatibility"].startswith("INCOMPATIBLE")
    assert report["action_in_projection"] == [32, 1024]
    assert report["action_out_projection"] == [1024, 32]
    assert report["state_projection_present"] is False
    assert report["weights_restored"] is False


@pytest.mark.parametrize(
    "failure",
    [
        "missing_tokenizer",
        "corrupt_tokenizer",
        "norm_stats",
        "tensor_metadata",
        "symlink",
    ],
)
def test_fr3_checkpoint_asset_failure_injection(
    fr3_checkpoint_assets, tmp_path, failure
):
    from rlinf.data.datasets.openpi.fr3_training import inspect_checkpoint_assets

    root = tmp_path / "assets"
    shutil.copytree(
        fr3_checkpoint_assets, root, ignore=shutil.ignore_patterns("source")
    )
    tokenizer = root / "big_vision/paligemma_tokenizer.model"
    if failure == "missing_tokenizer":
        tokenizer.unlink()
    elif failure == "corrupt_tokenizer":
        data = bytearray(tokenizer.read_bytes())
        data[-1] ^= 1
        tokenizer.write_bytes(data)
    elif failure == "norm_stats":
        (
            root / "openpi-assets/checkpoints/pi05_base/assets/franka/norm_stats.json"
        ).write_text("{}")
    elif failure == "tensor_metadata":
        (
            root
            / "openpi-assets/checkpoints/pi05_base/params/array_metadatas/process_0"
        ).write_text("{}")
    else:
        tokenizer.unlink()
        tokenizer.symlink_to(
            fr3_checkpoint_assets / "big_vision/paligemma_tokenizer.model"
        )
    with pytest.raises((ValueError, OSError), match="[Aa]sset|[Ss]ymlink|[Mm]issing"):
        inspect_checkpoint_assets(root)


def test_fr3_checkpoint_native_tokenizer_fixture_has_no_truncation(
    fr3_checkpoint_assets, fr3_openpi_dataset, monkeypatch
):
    from rlinf.data.datasets.openpi.fr3_training import tokenize_fr3_fixture

    monkeypatch.setenv("OPENPI_DATA_HOME", str(fr3_checkpoint_assets))
    state = np.zeros(42, dtype=np.float32)
    original = state.copy()
    probe = tokenize_fr3_fixture(
        fr3_checkpoint_assets,
        state,
        "synthetic tokenizer schema probe only",
        scope="synthetic_math_only",
    )
    assert probe["token_count"] == 184
    assert probe["vocabulary_size"] == 257152
    assert probe["tokens"].shape == probe["mask"].shape == (200,)
    assert probe["tokens"].dtype == np.int32 and probe["mask"].dtype == np.bool_
    assert not probe["truncated"] and not probe["checkpoint_semantics_qualified"]
    assert np.array_equal(original, state)
    assert (probe["tokens"][~probe["mask"]] == 0).all()
    assert fr3_openpi_dataset[0]["prompt"] is None


@pytest.mark.parametrize(
    "failure",
    [
        "scope",
        "dimension19",
        "dimension32",
        "dtype",
        "nonfinite",
        "range",
        "instruction",
        "truncation",
        "cache",
    ],
)
def test_fr3_checkpoint_tokenizer_rejects_unqualified_inputs(
    fr3_checkpoint_assets, fr3_openpi_dataset, monkeypatch, failure
):
    from rlinf.data.datasets.openpi.fr3_training import tokenize_fr3_fixture

    monkeypatch.setenv("OPENPI_DATA_HOME", str(fr3_checkpoint_assets))
    state, prompt, scope = (
        np.zeros(42, dtype=np.float32),
        "synthetic probe",
        "synthetic_math_only",
    )
    if failure == "scope":
        scope = "qualified_training"
    elif failure.startswith("dimension"):
        state = np.zeros(int(failure[-2:]), dtype=np.float32)
    elif failure == "dtype":
        state = state.astype(np.float64)
    elif failure == "nonfinite":
        state[0] = np.nan
    elif failure == "range":
        state[0] = 2
    elif failure == "instruction":
        prompt = ""
    elif failure == "truncation":
        prompt = "long synthetic instruction " * 300
    else:
        monkeypatch.setenv("OPENPI_DATA_HOME", "/tmp/unqualified-fr3-cache")
    with pytest.raises(ValueError):
        tokenize_fr3_fixture(fr3_checkpoint_assets, state, prompt, scope=scope)


@pytest.mark.parametrize("horizon", [1, 4, 20, 50])
def test_fr3_checkpoint_request_target_proposal_preserves_values_and_masks(
    fr3_openpi_export, horizon
):
    from rlinf.data.datasets.openpi.fr3 import FR3DatasetConfig, FR3OfflineDataset
    from rlinf.data.datasets.openpi.fr3_training import (
        build_fr3_request_target_proposal,
    )

    ds = FR3OfflineDataset(
        fr3_openpi_export, config=FR3DatasetConfig(action_horizon=horizon)
    )
    before = _fr3_dataset_hashes(ds.root)
    for index in range(len(ds)):
        proposal = build_fr3_request_target_proposal(ds, index)
        native = ds[index]
        np.testing.assert_array_equal(proposal["native_state"], native["state"])
        assert proposal["native_state"].shape == (42,)
        assert proposal["proposed_actions"].shape == (horizon, 32)
        np.testing.assert_array_equal(
            proposal["proposed_actions"][:, :7], native["recorded_controller_requests"]
        )
        assert (proposal["proposed_actions"][:, 7:] == 0).all()
        mask = proposal["proposed_loss_mask"]
        assert mask.dtype == np.bool_
        np.testing.assert_array_equal(
            mask[:, :7], np.repeat((~native["actions_is_pad"])[:, None], 7, axis=1)
        )
        assert not mask[:, 7:].any()
        assert mask.sum() == 7 * min(horizon, 4 - index % 4)
        assert proposal["model_training_target"] is None
        assert proposal["normalization_applied"] is False
        assert proposal["native_sft_mask_supported"] is False
        assert proposal["physical_execution"] is False
        proposal["proposed_actions"][0, 0] = 123
        assert ds[index]["recorded_controller_requests"][0, 0] != 123
    assert _fr3_dataset_hashes(ds.root) == before


def test_fr3_checkpoint_full_window_policy_and_execution_guard(fr3_openpi_dataset):
    from rlinf.data.datasets.openpi.fr3_training import (
        build_fr3_request_target_proposal,
        require_fr3_training_ready,
    )

    assert build_fr3_request_target_proposal(
        fr3_openpi_dataset, 0, require_full_window=True
    )["proposed_loss_mask"][:, :7].all()
    with pytest.raises(ValueError, match="Missing future actions"):
        build_fr3_request_target_proposal(
            fr3_openpi_dataset, 1, require_full_window=True
        )
    with pytest.raises(ValueError, match="audited FR3OfflineDataset"):
        build_fr3_request_target_proposal({}, 0)
    with pytest.raises(ValueError, match="CHECKPOINT_UNVERIFIED"):
        require_fr3_training_ready()


def test_fr3_checkpoint_future_collection_requirements_remain_missing(
    fr3_openpi_dataset,
):
    from rlinf.data.datasets.openpi.fr3_training import assess_fr3_training_requirements

    before = fr3_openpi_dataset.metadata_document()
    report = assess_fr3_training_requirements(fr3_openpi_dataset)
    missing = report["missing_episode_counts"]
    for field in (
        "original_instruction",
        "task_id_version",
        "episode_split",
        "normalization_provenance",
        "camera_configuration",
        "collection_session",
    ):
        assert missing[field] == 2
    assert report["action_schema_version"] == report["observation_schema_version"] == 1
    assert len(report["action_authority_sessions"]) == 2
    assert report["training_readiness"] == "UNQUALIFIED_SYNTHETIC_ONLY"
    assert fr3_openpi_dataset.metadata_document() == before


def test_fr3_checkpoint_cli_rejects_overlapping_output(fr3_openpi_export, monkeypatch):
    import sys

    from toolkits.realworld_check.qualify_fr3_checkpoint import main

    before = _fr3_dataset_hashes(fr3_openpi_export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify",
            "--export",
            str(fr3_openpi_export),
            "--assets",
            "/tmp/assets",
            "--output",
            str(fr3_openpi_export / "nested"),
        ],
    )
    assert main() == 1
    assert _fr3_dataset_hashes(fr3_openpi_export) == before


def test_fr3_checkpoint_missing_assets_produce_diagnostic(tmp_path):
    from toolkits.realworld_check.qualify_fr3_checkpoint import qualify_fr3_checkpoint

    pytest.importorskip("openpi")
    report = qualify_fr3_checkpoint(tmp_path / "dataset", tmp_path / "assets")
    assert not report["offline_inspection_complete"]
    assert report["diagnostics"] and report["diagnostics"][0]["message"]
    assert report["decisions"]["checkpoint_compatibility"] == "CHECKPOINT_UNVERIFIED"
    assert report["physical_execution"] is False


def test_fr3_checkpoint_end_to_end_offline_container_probe(
    fr3_checkpoint_assets, fr3_openpi_export, monkeypatch
):
    from toolkits.realworld_check.qualify_fr3_checkpoint import qualify_fr3_checkpoint

    monkeypatch.setenv("OPENPI_DATA_HOME", str(fr3_checkpoint_assets))
    before = _fr3_dataset_hashes(fr3_openpi_export)
    report = qualify_fr3_checkpoint(fr3_openpi_export, fr3_checkpoint_assets)
    assert report["offline_inspection_complete"], report["diagnostics"]
    assert report["dataset"]["episodes"] == 2 and report["dataset"]["transitions"] == 8
    assert report["temporal"]["full_windows"] == 0
    assert report["temporal"]["padded_future_steps"] == 140
    assert report["temporal"]["proposed_valid_loss_elements"] == 140
    assert not report["temporal"]["native_sft_consumes_padding_mask"]
    assert report["batch_structure"]["native_state_shape"] == [1, 42]
    assert not report["batch_structure"]["state_matches_default_spec"]
    assert report["decisions"]["training_data_readiness"].startswith("UNQUALIFIED")
    assert report["decisions"]["recap_readiness"].startswith("UNQUALIFIED")
    assert not report["training_run"] and not report["model_constructed"]
    assert _fr3_dataset_hashes(fr3_openpi_export) == before


def test_fr3_checkpoint_cli_forbids_network_models_and_hardware(
    fr3_checkpoint_assets, fr3_openpi_export, tmp_path
):
    import os
    import subprocess
    import sys

    program = r"""
import builtins, subprocess, socket, psutil, ray
original_import = builtins.__import__
original_popen = subprocess.Popen
def forbidden(*args, **kwargs):
    raise AssertionError('network, model construction or hardware launch forbidden')
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split('.')[0] in {'rclpy','rospy','franky','pyrealsense2','hid','hidapi','spnav','pyspacemouse','pymodbus'}:
        raise AssertionError('hardware SDK import: '+name)
    return original_import(name, globals, locals, fromlist, level)
def checked_popen(args, *pos, **kwargs):
    if (args == 'lscpu' or args == ['uname', '-p']) and not kwargs.get('shell', False):
        return original_popen(args, *pos, **kwargs)
    return forbidden(args, *pos, **kwargs)
builtins.__import__ = guarded
import numpy.testing
from rlinf.envs.real.fr3 import config as offline_schema
subprocess.Popen = checked_popen
psutil.process_iter = forbidden
socket.socket.connect = forbidden
ray.init = forbidden
from openpi.models.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi.pi0 import Pi0
import openpi.shared.download as download
Pi0Config.create = forbidden
Pi0.__init__ = forbidden
download._download_fsspec = forbidden
from toolkits.realworld_check.qualify_fr3_checkpoint import main
raise SystemExit(main())
"""
    output = tmp_path / "checkpoint-report"
    before = _fr3_dataset_hashes(fr3_openpi_export)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            "--export",
            str(fr3_openpi_export),
            "--assets",
            str(fr3_checkpoint_assets),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "OPENPI_DATA_HOME": str(fr3_checkpoint_assets),
            "JAX_PLATFORMS": "cpu",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_CACHE": str(tmp_path / "hf-cache"),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "qualification.json").read_text())
    assert report["offline_inspection_complete"]
    assert report["physical_command_authorization"] == "NOT_GRANTED"
    assert report["model_constructed"] is False
    assert _fr3_dataset_hashes(fr3_openpi_export) == before
