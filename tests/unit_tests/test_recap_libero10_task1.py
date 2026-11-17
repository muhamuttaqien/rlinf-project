"""CPU-only contract tests for the isolated LIBERO-10 Task 1 RECAP setup."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from hydra import compose, initialize_config_dir

from toolkits.recap_libero10_task1_collect import DEFAULT_DATA_ROOT, SPLITS, episode_frames, features
from toolkits.recap_libero10_task1_preflight import TAG, check_datasets

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/offline_rl/config/experiments/libero10_task1"


class LongTask1Tests(unittest.TestCase):
    def test_isolated_splits_and_frame_schema(self):
        self.assertEqual(DEFAULT_DATA_ROOT.name, "recap_libero10_task1")
        self.assertEqual([name for name, _ in SPLITS.values()], [
            "libero10_task1_sft", "libero10_task1_train", "libero10_task1_eval"
        ])
        starts = [seed for _, seed in SPLITS.values()]
        self.assertEqual(len(starts), len(set(starts)))
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        action = np.arange(7, dtype=np.float32)
        rows = episode_frames(
            [(img, img, np.zeros(8, dtype=np.float32), action)] * 2,
            success=True,
            task="task one instruction",
        )
        self.assertEqual(len(rows), 2)
        self.assertFalse(bool(rows[0]["done"][0]))
        self.assertTrue(bool(rows[-1]["done"][0]))
        self.assertTrue(bool(rows[-1]["is_success"][0]))
        self.assertEqual(features((8, 8, 3))["actions"]["shape"], (7,))

    def test_config_tags_and_training_parity(self):
        def read(name):
            return yaml.safe_load((CONFIG / f"{name}.yaml").read_text())

        ret = read("recap_compute_returns")
        critic = read("recap_value_model_sft")
        adv = read("recap_compute_advantages")
        policy = read("cfg_rl_openpi")
        self.assertEqual(ret["data"]["tag"], TAG)
        self.assertEqual(critic["data"]["tag"], TAG)
        self.assertEqual(adv["advantage"]["tag"], TAG)
        self.assertEqual(adv["advantage"]["returns_tag"], TAG)
        self.assertEqual(policy["data"]["advantage_tag"], TAG)
        self.assertEqual(critic["runner"]["max_steps"], 8000)
        self.assertEqual(critic["actor"]["optim"]["total_training_steps"], 8000)
        self.assertEqual(policy["runner"]["max_steps"], 3000)
        self.assertEqual(policy["actor"]["optim"]["total_training_steps"], 30000)
        self.assertEqual(policy["actor"]["optim"]["lr_warmup_steps"], 5000)
        self.assertEqual(policy["actor"]["global_batch_size"], 512)
        self.assertEqual(critic["actor"]["global_batch_size"], 256)
        self.assertTrue(critic["data"]["train_data_paths"][0]["dataset_path"].endswith("libero10_task1_sft"))
        self.assertTrue(adv["data"]["train_data_paths"][0]["dataset_path"].endswith("libero10_task1_train"))
        self.assertEqual([p["type"] for p in policy["data"]["train_data_paths"]], ["sft", "rollout"])
        self.assertEqual(policy["actor"]["model"]["openpi"]["guidance_type"], "positive")
        self.assertEqual(policy["actor"]["model"]["model_path"].split("/")[-1], "pi05_libero_sft_rlinf")

    def test_hydra_composition(self):
        config_root = ROOT / "examples/offline_rl/config"
        with patch.dict(os.environ, {"REPO_PATH": str(ROOT)}):
            with initialize_config_dir(version_base="1.1", config_dir=str(config_root)):
                for name in ("recap_compute_returns", "recap_value_model_sft",
                             "recap_compute_advantages", "cfg_rl_openpi"):
                    with self.subTest(config=name):
                        cfg = compose(config_name=f"experiments/libero10_task1/{name}")
                        self.assertNotIn("experiments", cfg)
                        self.assertIn("data", cfg)
                        if name in ("recap_value_model_sft", "cfg_rl_openpi"):
                            self.assertIn("actor", cfg)

    def test_preflight_rejects_missing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems = check_datasets(Path(tmp))
        self.assertEqual(len(problems), 3)


if __name__ == "__main__":
    unittest.main()
