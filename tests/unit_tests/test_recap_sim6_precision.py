# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""SIM-6 regression: BF16 image inputs must not break OpenPI grid_sample."""

import pytest
import torch

from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
    Observation,
    OpenPi0ForCFGActionPrediction,
)


@pytest.mark.parametrize("train", [True, False])
def test_bf16_inputs_use_fp32_image_preprocessing(train):
    # The production run enters this function inside a BF16 autocast context.
    batch = 2
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    images = {
        key: torch.rand(batch, 3, 224, 224, dtype=torch.bfloat16)
        for key in keys
    }
    observation = Observation(
        images=images,
        image_masks={key: torch.ones(batch, dtype=torch.bool) for key in keys},
        state=torch.randn(batch, 8, dtype=torch.bfloat16),
        tokenized_prompt=torch.ones(batch, 8, dtype=torch.int64),
        tokenized_prompt_mask=torch.ones(batch, 8, dtype=torch.bool),
        tokenized_positive_guidance_prompt=torch.ones(batch, 8, dtype=torch.int64),
        tokenized_positive_guidance_prompt_mask=torch.ones(batch, 8, dtype=torch.bool),
        tokenized_negative_guidance_prompt=torch.ones(batch, 8, dtype=torch.int64),
        tokenized_negative_guidance_prompt_mask=torch.ones(batch, 8, dtype=torch.bool),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = OpenPi0ForCFGActionPrediction._preprocess_observation(
            None, observation, train=train
        )

    processed_images = output[0]
    assert len(processed_images) == 3
    assert all(image.dtype == torch.float32 for image in processed_images)
    assert all(image.shape == (batch, 3, 224, 224) for image in processed_images)
    # Do not mutate the caller's original observations.
    assert all(image.dtype == torch.bfloat16 for image in observation.images.values())
