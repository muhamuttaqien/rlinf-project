"""Isolated checks for interrupted LIBERO comparison recovery (no GPU required)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np

from toolkits.recap_libero_compare_videos import (
    _recover_policy, _run_evaluator, _valid_existing_episode,
)


def _record_episode(folder: Path, episode: int, seed: int, with_wrist=True):
    folder.mkdir(parents=True, exist_ok=True)
    frames = [np.full((32, 32, 3), k * 24, dtype=np.uint8) for k in range(4)]
    imageio.mimwrite(folder / f"task0_episode{episode}_success.mp4", frames, fps=3)
    if with_wrist:
        imageio.mimwrite(folder / f"wrist_task0_episode{episode}_success.mp4", frames, fps=3)
    np.savez_compressed(
        folder / f"state_task0_episode{episode}.npz",
        states=np.zeros((len(frames), 8), dtype=np.float32),
        prompt=np.array("place the object"), seed=np.array(seed),
        episode=np.array(episode),
    )


class TestResume(unittest.TestCase):
    def test_complete_recording_is_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "baseline"
            _record_episode(folder, 1, 7)
            self.assertTrue(_valid_existing_episode(folder, 1, 7, True))
            missing = _recover_policy(folder, "baseline", 1, 7, True, root / "backup")
            self.assertEqual(missing, [])
            self.assertTrue((folder / "task0_episode1_success.mp4").exists())

    def test_partial_recording_is_preserved_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "recap"
            _record_episode(folder, 1, 7, with_wrist=False)
            self.assertFalse(_valid_existing_episode(folder, 1, 7, True))
            missing = _recover_policy(folder, "recap", 1, 7, True, root / "backup")
            self.assertEqual(missing, [1])
            self.assertTrue((root / "backup/recap/task0_episode1_success.mp4").exists())
            self.assertTrue((root / "backup/recap/state_task0_episode1.npz").exists())
            self.assertFalse((folder / "task0_episode1_success.mp4").exists())

    def test_mismatched_seed_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "baseline"
            _record_episode(folder, 1, 999)
            self.assertFalse(_valid_existing_episode(folder, 1, 7, True))

    def test_only_missing_episode_numbers_are_sent_to_evaluator(self):
        with patch("toolkits.recap_libero_compare_videos.subprocess.run") as run:
            _run_evaluator(Path("/repo"), Path("/sft"), None, Path("/out"), 20, 7, True, [4, 8])
        command = run.call_args.args[0]
        i = command.index("--episode-indices")
        self.assertEqual(command[i + 1], "4,8")
        self.assertIn("--value-observations", command)


if __name__ == "__main__":
    unittest.main()
