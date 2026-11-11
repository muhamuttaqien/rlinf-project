"""Unit checks for the optional NVIDIA count fallback used by Goal RECAP.

This does not initialize Ray, allocate GPUs, or run training.
"""

import os
import unittest
from unittest.mock import patch

from ray._private.accelerators.nvidia_gpu import NvidiaGPUAcceleratorManager
from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

OVERRIDE = "RLINF_NVIDIA_GPU_COUNT_OVERRIDE"


class RayGpuFallbackTests(unittest.TestCase):
    def test_no_override_preserves_ray_zero(self):
        with patch.object(NvidiaGPUAcceleratorManager, "get_current_node_num_accelerators", return_value=0):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(OVERRIDE, None)
                self.assertEqual(NvidiaGPUManager.get_num_devices(), 0)

    def test_explicit_fallback_only_when_ray_reports_zero(self):
        with patch.object(NvidiaGPUAcceleratorManager, "get_current_node_num_accelerators", return_value=0):
            with patch.dict(os.environ, {OVERRIDE: "4"}):
                self.assertEqual(NvidiaGPUManager.get_num_devices(), 4)

    def test_ray_detection_takes_precedence(self):
        with patch.object(NvidiaGPUAcceleratorManager, "get_current_node_num_accelerators", return_value=2):
            with patch.dict(os.environ, {OVERRIDE: "4"}):
                self.assertEqual(NvidiaGPUManager.get_num_devices(), 2)

    def test_invalid_override_fails_fast(self):
        with patch.object(NvidiaGPUAcceleratorManager, "get_current_node_num_accelerators", return_value=0):
            for bad in ("", "0", "-1", "four", "3.5"):
                with self.subTest(value=bad), patch.dict(os.environ, {OVERRIDE: bad}):
                    with self.assertRaises(ValueError):
                        NvidiaGPUManager.get_num_devices()


if __name__ == "__main__":
    unittest.main()
