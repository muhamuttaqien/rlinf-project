# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused regression tests for state-conditioned StackCube Pi0.5 SFT."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "examples/sft/config"
VISION_YAML = CONFIG_ROOT / "maniskill_stackcube_panda_sft_openpi_pi05.yaml"
STATE_YAML = CONFIG_ROOT / "maniskill_stackcube_panda_sft_openpi_pi05_state.yaml"


def test_state_sft_is_separate_and_preserves_controlled_training_settings():
    vision = OmegaConf.load(VISION_YAML)
    state = OmegaConf.load(STATE_YAML)

    assert vision.actor.model.openpi.config_name == "pi05_stackcube_panda_joint"
    assert vision.actor.model.openpi.discrete_state_input is False
    assert state.actor.model.openpi.config_name == "pi05_stackcube_panda_joint_state"
    assert state.actor.model.openpi.discrete_state_input is True

    assert state.runner.logger.project_name == vision.runner.logger.project_name
    assert state.runner.logger.experiment_name != vision.runner.logger.experiment_name
    for key in ("max_steps", "save_interval"):
        assert state.runner[key] == vision.runner[key]
    for key in ("global_batch_size", "micro_batch_size", "seed"):
        assert state.actor[key] == vision.actor[key]
    for key in ("lr", "lr_warmup_steps", "lr_scheduler", "min_lr"):
        assert state.actor.optim[key] == vision.actor.optim[key]
    assert state.actor.model.openpi.action_horizon == 10
    assert state.actor.model.openpi.num_images_in_input == 1
    assert state.actor.model.action_dim == 8
    assert state.actor.model.openpi.model_action_dim == 32
    assert state.actor.model.openpi.use_rlt is False
    assert (
        state.actor.model.openpi_data.norm_stats_path
        == vision.actor.model.openpi_data.norm_stats_path
    )


def test_state_flag_matches_registered_openpi_config_and_data_contract():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    for name, enabled in (
        ("pi05_stackcube_panda_joint", False),
        ("pi05_stackcube_panda_joint_state", True),
    ):
        cfg = get_openpi_config(name)
        assert cfg.model.pi05 is True
        assert cfg.model.discrete_state_input is enabled
        assert cfg.model.action_horizon == 10
        assert cfg.model.action_dim == 32
        assert cfg.data.output_action_dim == 8
        assert cfg.data.use_wrist_image is False
        assert cfg.data.extra_delta_transform is False


def test_upstream_tokenizer_receives_state_only_for_state_conditioned_variant(
    monkeypatch,
):
    pytest.importorskip("openpi")
    from openpi import transforms
    import openpi.training.config as openpi_training_config
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    # Avoid downloading a tokenizer in this targeted CPU unit test.
    calls = []

    class FakeTokenizer:
        def __init__(self, max_len):
            assert max_len == 200

        def tokenize(self, prompt, state):
            calls.append((prompt, None if state is None else state.copy()))
            return (
                np.array([1, 2, 3], dtype=np.int32),
                np.array([True, True, True], dtype=np.bool_),
            )

    monkeypatch.setattr(
        openpi_training_config._tokenizer, "PaligemmaTokenizer", FakeTokenizer
    )
    state = np.linspace(-0.7, 0.7, 9, dtype=np.float32)
    for config_name, enabled in (
        ("pi05_stackcube_panda_joint", False),
        ("pi05_stackcube_panda_joint_state", True),
    ):
        registered = get_openpi_config(config_name)
        group = openpi_training_config.ModelTransformFactory()(registered.model)
        tokenize = next(
            transform for transform in group.inputs
            if isinstance(transform, transforms.TokenizePrompt)
        )
        assert tokenize.discrete_state_input is enabled
        result = tokenize({"prompt": "stack cubes", "state": state.copy()})
        assert result["tokenized_prompt"].shape == (3,)
        prompt, received_state = calls[-1]
        assert prompt == "stack cubes"
        if enabled:
            np.testing.assert_array_equal(received_state, state)
        else:
            assert received_state is None


def test_training_loader_rejects_model_tokenizer_state_mismatch():
    pytest.importorskip("openpi")
    from rlinf.data.datasets.openpi.official_sft_data_loader import (
        _validate_openpi_model_shape,
    )
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    cfg = OmegaConf.load(STATE_YAML).actor.model
    upstream = get_openpi_config("pi05_stackcube_panda_joint_state")
    _validate_openpi_model_shape(cfg, upstream)
    cfg.openpi.discrete_state_input = False
    with pytest.raises(ValueError, match="state conditioning mismatch"):
        _validate_openpi_model_shape(cfg, upstream)


def test_evaluator_accepts_state_config_and_rejects_wrong_state_flag(tmp_path):
    from scripts.evaluate_stackcube_pi05 import make_model_config

    base = CONFIG_ROOT / "model/pi0_5.yaml"
    checkpoint = tmp_path / "weights"
    stats = tmp_path / "norm_stats.json"
    model_cfg, _ = make_model_config(
        STATE_YAML, base, checkpoint, stats
    )
    assert model_cfg.openpi.config_name == "pi05_stackcube_panda_joint_state"
    assert model_cfg.openpi.discrete_state_input is True
    assert str(model_cfg.model_path) == str(checkpoint.resolve())

    text = STATE_YAML.read_text(encoding="utf-8")
    mismatched = tmp_path / "mismatched.yaml"
    mismatched.write_text(
        text.replace(
            "discrete_state_input: True",
            "discrete_state_input: False",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="State conditioning mismatch"):
        make_model_config(mismatched, base, checkpoint, stats)
