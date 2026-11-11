"""CPU-only tests for task-specific LIBERO-Goal Task 0 RECAP setup."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from toolkits.recap_libero_goal_collect import episode_frames, features, SPLITS
from toolkits.recap_libero_goal_preflight import check_datasets
from examples.offline_rl.advantage_labeling.recap.process.compute_returns import (
    parse_episode_success, compute_returns_for_episode,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/offline_rl/config/experiments/libero_goal_task0"
TAG = "libero_goal_task0_recap"


class GoalTask0Tests(unittest.TestCase):
    def test_frames_record_executed_actions_and_success(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        action = np.arange(7, dtype=np.float32)
        obs = [(img, img, np.zeros(8, np.float32), action)] * 3
        frames = episode_frames(obs, success=False, task="move the bowl")
        self.assertEqual(len(frames), 3)
        self.assertFalse(bool(frames[-1]["is_success"][0]))
        self.assertTrue(bool(frames[-1]["done"][0]))
        self.assertFalse(bool(frames[0]["done"][0]))
        np.testing.assert_array_equal(frames[1]["actions"], action)

    def test_invalid_actions_rejected(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "action"):
            episode_frames([(img, img, np.zeros(8), np.zeros(8))], False, "task")

    def test_success_flag_normalization(self):
        self.assertFalse(parse_episode_success([False]))
        self.assertTrue(parse_episode_success([True]))
        self.assertFalse(parse_episode_success(np.array([0], dtype=np.bool_)))
        self.assertFalse(parse_episode_success(False))
        for invalid in ([False, True], [], None, "False", 2):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_episode_success(invalid)

    def test_failure_reward_is_negative(self):
        successes, _ = compute_returns_for_episode(3, True, 1.0, -300)
        failures, rewards = compute_returns_for_episode(3, False, 1.0, -300)
        self.assertEqual(float(rewards[-1]), -300.0)
        self.assertLess(float(failures[0]), float(successes[0]))

    def test_goal_configs_are_task_isolated(self):
        with (CONFIG / "recap_compute_returns.yaml").open() as stream:
            ret = yaml.safe_load(stream)
        with (CONFIG / "recap_value_model_sft.yaml").open() as stream:
            critic = yaml.safe_load(stream)
        with (CONFIG / "recap_compute_advantages.yaml").open() as stream:
            adv = yaml.safe_load(stream)
        with (CONFIG / "cfg_rl_openpi.yaml").open() as stream:
            policy = yaml.safe_load(stream)
        self.assertEqual(ret["data"]["tag"], TAG)
        self.assertEqual(critic["data"]["tag"], TAG)
        self.assertEqual(adv["advantage"]["tag"], TAG)
        self.assertEqual(adv["advantage"]["returns_tag"], TAG)
        self.assertEqual(policy["data"]["advantage_tag"], TAG)
        self.assertEqual(critic["actor"]["optim"]["total_training_steps"], 8000)
        self.assertEqual(policy["actor"]["optim"]["total_training_steps"], 3000)
        self.assertEqual(policy["runner"]["max_steps"], 3000)
        self.assertTrue(policy["actor"]["model"]["model_path"].endswith("pi05_libero_sft_rlinf"))
        self.assertIn("REPLACE_WITH_GOAL_TASK0_CRITIC", adv["advantage"]["value_checkpoint"])
        sft, train, evaluation = (Path(d["dataset_path"]).name for d in ret["data"]["train_data_paths"])
        self.assertEqual((sft, train, evaluation), tuple(name for name, _ in SPLITS.values()))
        self.assertEqual(features((8, 8, 3))["actions"]["shape"], (7,))

    def test_preflight_blocks_missing_goal_datasets(self):
        with tempfile.TemporaryDirectory() as temp:
            result = check_datasets(Path(temp))
        self.assertEqual(len(result), 3)
        self.assertTrue(all("missing complete Goal Task 0 dataset" in x for x in result))


if __name__ == "__main__":
    unittest.main()
