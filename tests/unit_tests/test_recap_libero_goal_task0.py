"""CPU-only tests for task-specific LIBERO-Goal Task 0 RECAP setup."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir

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

    def test_mixed_value_critic_configs_have_isolated_train_only_stats(self):
        # The completed 8k success-only critic remains a separate baseline.
        with (CONFIG / "recap_value_model_sft.yaml").open() as stream:
            legacy = yaml.safe_load(stream)
        self.assertTrue(legacy["data"]["train_data_paths"][0]["dataset_path"].endswith(
            "libero_goal_task0_sft"
        ))
        for variant, expected_steps in (("pilot", 1000), ("full", 8000)):
            with self.subTest(variant=variant):
                with (CONFIG / f"recap_value_model_mixed_{variant}.yaml").open() as stream:
                    cfg = yaml.safe_load(stream)
                self.assertEqual(cfg["data"]["tag"], TAG)
                train_paths = cfg["data"]["train_data_paths"]
                self.assertEqual(len(train_paths), 1)
                self.assertEqual(train_paths[0]["type"], "rollout")
                self.assertTrue(train_paths[0]["dataset_path"].endswith(
                    "libero_goal_task0_train"
                ))
                self.assertEqual(len(cfg["data"]["eval_data_paths"]), 1)
                self.assertTrue(cfg["data"]["eval_data_paths"][0]["dataset_path"].endswith(
                    "libero_goal_task0_eval"
                ))
                # Worker derives global return_min/max from TRAIN stats.json.
                self.assertIsNone(cfg["data"].get("return_min"))
                self.assertIsNone(cfg["data"].get("return_max"))
                self.assertTrue(cfg["data"]["normalize_to_minus_one_zero"])
                self.assertEqual(cfg["actor"]["model"]["action_dim"], 7)
                self.assertEqual(cfg["actor"]["micro_batch_size"], 32)
                self.assertEqual(cfg["actor"]["global_batch_size"], 256)
                self.assertEqual(cfg["actor"]["optim"]["total_training_steps"], 8000)
                self.assertEqual(cfg["actor"]["optim"]["lr_warmup_steps"], 500)
                self.assertEqual(cfg["runner"]["max_steps"], expected_steps)
                self.assertEqual(cfg["runner"]["save_interval"], 500)
                self.assertEqual(cfg["cluster"]["component_placement"]["actor,env,rollout"], "2")

    def test_hydra_configs_use_root_package(self):
        # YAML parsing alone does not detect Hydra's implicit package prefix:
        # nested 'experiments/libero_goal_task0' would hide cfg.data.
        config_root = ROOT / "examples/offline_rl/config"
        with patch.dict(os.environ, {"REPO_PATH": str(ROOT)}):
            with initialize_config_dir(version_base="1.1", config_dir=str(config_root)):
                names = (
                    "recap_compute_returns",
                    "recap_value_model_sft",
                    "recap_value_model_mixed_pilot",
                    "recap_value_model_mixed_full",
                    "recap_compute_advantages",
                    "cfg_rl_openpi",
                )
                for name in names:
                    with self.subTest(name=name):
                        cfg = compose(config_name=f"experiments/libero_goal_task0/{name}")
                        self.assertIn("data", cfg)
                        self.assertNotIn("experiments", cfg)
                        if name == "recap_compute_advantages":
                            self.assertIn("advantage", cfg)
                        if name in (
                            "recap_value_model_sft",
                            "recap_value_model_mixed_pilot",
                            "recap_value_model_mixed_full",
                            "cfg_rl_openpi",
                        ):
                            self.assertIn("actor", cfg)

    def test_preflight_blocks_missing_goal_datasets(self):
        with tempfile.TemporaryDirectory() as temp:
            result = check_datasets(Path(temp))
        self.assertEqual(len(result), 3)
        self.assertTrue(all("missing complete Goal Task 0 dataset" in x for x in result))


if __name__ == "__main__":
    unittest.main()
